import os
import torch


import jiant.proj.main.modeling.model_setup as jiant_model_setup
import jiant.proj.main.runner as jiant_runner
import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.metarunner as jiant_metarunner
import jiant.proj.main.components.evaluate as jiant_evaluate
import jiant.shared.initialization as initialization
import jiant.shared.distributed as distributed
import jiant.shared.model_setup as model_setup
import jiant.utils.torch_utils as torch_utils
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    jiant_task_container_config_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_transformers", type=str)

    # === Running Setup === #
    do_train = zconf.attr(action="store_true")
    do_val = zconf.attr(action="store_true")
    do_save = zconf.attr(action="store_true")
    write_val_preds = zconf.attr(action="store_true")
    write_test_preds = zconf.attr(action="store_true")
    eval_every_steps = zconf.attr(type=int, default=0)
    save_every_steps = zconf.attr(type=int, default=0)
    save_checkpoint_every_steps = zconf.attr(type=int, default=0)
    no_improvements_for_n_evals = zconf.attr(type=int, default=0)
    delete_checkpoint_if_done = zconf.attr(action="store_true")
    force_overwrite = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # Specialized config
    no_cuda = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    fp16_opt_level = zconf.attr(default="O1", type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default="", type=str)
    server_port = zconf.attr(default="", type=str)

    # New for result writing
    val_jsonl = zconf.attr(action="store_true")
    args_jsonl = zconf.attr(action="store_true")
    custom_best_name = zconf.attr(default="", type=str)
    custom_checkpoint_name = zconf.attr(default="", type=str)
    custom_logger_post = zconf.attr(default="", type=str)
    extract_exp_name_valpreds = zconf.attr(action="store_true")


@zconf.run_config
class ResumeConfiguration(zconf.RunConfig):
    checkpoint_path = zconf.attr(type=str)


def setup_runner(
    args: RunConfiguration,
    jiant_task_container: container_setup.JiantTaskContainer,
    quick_init_out,
    verbose: bool = True,
) -> jiant_runner.JiantRunner:
    """Setup jiant model, optimizer, and runner, and return runner.

    Args:
        args (RunConfiguration): configuration carrying command line args specifying run params.
        jiant_task_container (container_setup.JiantTaskContainer): task and sampler configs.
        quick_init_out (QuickInitContainer): device (GPU/CPU) and logging configuration.
        verbose: If True, enables printing configuration info (to standard out).

    Returns:
        jiant_runner.JiantRunner

    """
    # TODO document why the distributed.only_first_process() context manager is being used here.
    with distributed.only_first_process(local_rank=args.local_rank):
        # load the model
        jiant_model = jiant_model_setup.setup_jiant_model(
            model_type=args.model_type,
            model_config_path=args.model_config_path,
            tokenizer_path=args.model_tokenizer_path,
            task_dict=jiant_task_container.task_dict,
            taskmodels_config=jiant_task_container.taskmodels_config,
        )
        jiant_model_setup.delegate_load_from_path(
            jiant_model=jiant_model, weights_path=args.model_path, load_mode=args.model_load_mode
        )
        jiant_model.to(quick_init_out.device)

    optimizer_scheduler = model_setup.create_optimizer(
        model=jiant_model,
        learning_rate=args.learning_rate,
        t_total=jiant_task_container.global_train_config.max_steps,
        warmup_steps=jiant_task_container.global_train_config.warmup_steps,
        warmup_proportion=None,
        optimizer_type=args.optimizer_type,
        verbose=verbose,
    )
    jiant_model, optimizer = model_setup.raw_special_model_setup(
        model=jiant_model,
        optimizer=optimizer_scheduler.optimizer,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        n_gpu=quick_init_out.n_gpu,
        local_rank=args.local_rank,
    )
    optimizer_scheduler.optimizer = optimizer
    rparams = jiant_runner.RunnerParameters(
        local_rank=args.local_rank,
        n_gpu=quick_init_out.n_gpu,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
    )
    runner = jiant_runner.JiantRunner(
        jiant_task_container=jiant_task_container,
        jiant_model=jiant_model,
        optimizer_scheduler=optimizer_scheduler,
        device=quick_init_out.device,
        rparams=rparams,
        log_writer=quick_init_out.log_writer,
    )
    return runner


def run_loop(args: RunConfiguration, checkpoint=None):
    is_resumed = checkpoint is not None
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    print(quick_init_out.n_gpu)
    with quick_init_out.log_writer.log_context():
        jiant_task_container = container_setup.create_jiant_task_container_from_json(
            jiant_task_container_config_path=args.jiant_task_container_config_path, verbose=True,
        )
        runner = setup_runner(
            args=args,
            jiant_task_container=jiant_task_container,
            quick_init_out=quick_init_out,
            verbose=True,
        )
        if is_resumed:
            runner.load_state(checkpoint["runner_state"])
            del checkpoint["runner_state"]

        # allow custom checkpoint name
        if args.custom_checkpoint_name:
            checkpoint_name = os.path.join(args.output_dir, f"{args.custom_checkpoint_name}.p")
        else:
            checkpoint_name = os.path.join(args.output_dir, "checkpoint.p")

        checkpoint_saver = jiant_runner.CheckpointSaver(
            metadata={"args": args.to_dict()},
            save_path=os.path.join(args.output_dir, checkpoint_name),
        )
        if args.do_train:
            metarunner = jiant_metarunner.JiantMetarunner(
                runner=runner,
                save_every_steps=args.save_every_steps,
                eval_every_steps=args.eval_every_steps,
                save_checkpoint_every_steps=args.save_checkpoint_every_steps,
                no_improvements_for_n_evals=args.no_improvements_for_n_evals,
                checkpoint_saver=checkpoint_saver,
                output_dir=args.output_dir,
                verbose=True,
                save_best_model=args.do_save,
                load_best_model=True,
                log_writer=quick_init_out.log_writer,
            )
            if is_resumed:
                metarunner.load_state(checkpoint["metarunner_state"])
                del checkpoint["metarunner_state"]
            metarunner.run_train_loop()

        if args.do_save:
            # allow custom best model name
            if args.custom_best_name:
                best_model_name = os.path.join(args.output_dir, f"{args.custom_best_name}.p")
            else:
                best_model_name = os.path.join(args.output_dir, "model.p")

            torch.save(
                torch_utils.get_model_for_saving(runner.jiant_model).state_dict(),
                best_model_name,
            )

        if args.do_val:
            val_results_dict = runner.run_val(
                task_name_list=runner.jiant_task_container.task_run_config.val_task_list,
                return_preds=args.write_val_preds,
            )
            jiant_evaluate.write_val_results(
                val_results_dict=val_results_dict,
                metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                output_dir=args.output_dir,
                verbose=True,
                val_jsonl=args.val_jsonl,
            )

            if args.args_jsonl:
                # match arguments with verbose results
                initialization.save_args(args, verbose=True, matched=True)

            if args.write_val_preds:
                if args.extract_exp_name_valpreds:
                    exp_name = os.path.basename(args.jiant_task_container_config_path).split(".")[0]
                    val_fname = f"val_preds_{exp_name}.p"
                else:
                    val_fname = "val_preds.p"
                jiant_evaluate.write_preds(
                    eval_results_dict=val_results_dict,
                    path=os.path.join(args.output_dir, val_fname),
                )
        else:
            assert not args.write_val_preds

        if args.write_test_preds:
            test_results_dict = runner.run_test(
                task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
            )
            jiant_evaluate.write_preds(
                eval_results_dict=test_results_dict,
                path=os.path.join(args.output_dir, "test_preds.p"),
            )

    if args.delete_checkpoint_if_done and args.save_checkpoint_every_steps:
        os.remove(os.path.join(args.output_dir, checkpoint_name))


def run_resume(args: ResumeConfiguration):
    checkpoint = torch.load(args.checkpoint_path)
    args = RunConfiguration.from_dict(checkpoint["metadata"]["args"])
    run_loop(args=args, checkpoint=checkpoint)


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    if mode == "run":
        run_loop(RunConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "continue":
        run_resume(ResumeConfiguration.default_run_cli(cl_args=cl_args))
    else:
        raise zconf.ModeLookupError(mode)


if __name__ == "__main__":
    main()
