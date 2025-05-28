import argparse
from torch.utils.data import DataLoader
from Source.data_handler import get_dataset_and_transforms, DataPoolManager
from Source.model_handler import ViTHandler
from Source.trainer import Trainer


def main_incremental(args: argparse.Namespace, exp_dir, logger) -> None:
    # Get the full dataset
    train_dataset, test_dataset, _ = get_dataset_and_transforms(args.dataset)
    data_manager = DataPoolManager(train_dataset, args)

    # Initialization Phase
    logger.info("Initialization Phase: Creating initial training set T_0")
    data_manager.initialize_T_s()
    initial_train_loader = DataLoader(
        data_manager.get_dataset_T_s(),
        batch_size=args.beta,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model_handler = ViTHandler(
        model_name=args.model_name,
        num_classes=len(train_dataset.classes),
        pretrained=args.pretrained,
        device=args.device,
        freeze_config_path=args.freeze_config_path,
        base_vit_config=args.base_vit_config
    )

    # Trainer initialization with method
    trainer = Trainer(
        model_handler=model_handler,
        data_manager=data_manager,
        test_loader=DataLoader(
            test_dataset,
            batch_size=args.beta,
            shuffle=False,
            pin_memory=True
        ),
        optimizer_params={
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
            "momentum": 0.9,
        },
        device=args.device,
        exp_dir=exp_dir,
        log_interval=args.log_interval,
        logger=logger,
        args=args
    )

    logger.info("Training on initial training set T_0")
    trainer.train_initial(initial_train_loader, num_steps=args.initial_training_steps)

    # Data Selection Phase
    logger.info("Starting Data Selection Phase")
    trainer.train_incremental()

    # Fine-tuning Phase
    if trainer.updates_remaining() > 0:
        logger.info("Starting Fine-tuning Phase")
        final_train_loader = DataLoader(
            data_manager.get_dataset_T_end(),
            batch_size=args.beta,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        trainer.fine_tune(final_train_loader)




