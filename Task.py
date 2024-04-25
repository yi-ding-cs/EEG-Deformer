from models.model_handler import DLModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import *


def LOSO(test_idx: list, subjects: list, experiment_ID, logs_name, args):
    pl.seed_everything(seed=args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    # load test data
    load_path = osp.join(args.load_path, 'data_{}_{}_{}'.format(args.data_format, args.dataset, args.label_type))
    data_test, label_test = load_data(load_path=load_path, load_idx=test_idx, concat=True)
    # load training data
    train_idx = [item for item in subjects if item not in test_idx]
    data_train, label_train = load_data(load_path=load_path, load_idx=train_idx, concat=True)
    train_idx, val_idx = get_validation_set(train_idx=np.arange(data_train.shape[0]), val_rate=args.val_rate, shuffle=True)
    data_val, label_val = data_train[val_idx], label_train[val_idx]
    data_train, label_train = data_train[train_idx], label_train[train_idx]
    # normalize the data
    data_train, data_val, data_test = normalize(train=data_train, val=data_val, test=data_test)

    print('Train:{} Val:{} Test:{}'.format(data_train.shape, data_val.shape, data_test.shape))
    # reorder the data for some models, e.g. TSception, LGGNet
    idx, _ = get_channel_info(load_path=load_path, graph_type=args.graph_type)
    # prepare dataloaders
    train_loader = prepare_data_for_training(data=data_train, label=label_train, idx=idx, batch_size=args.batch_size,
                                             shuffle=True)
    val_loader = prepare_data_for_training(data=data_val, label=label_val, idx=idx, batch_size=args.batch_size,
                                           shuffle=False)
    test_loader = prepare_data_for_training(data=data_test, label=label_test, idx=idx, batch_size=1000,
                                            shuffle=False)
    # train and test the model
    model = DLModel(config=args)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode='max')
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=experiment_ID, name=logs_name)
    # most basic trainer, uses good defaults (1 gpu)
    if args.mixed_precision:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback], precision='16-mixed'
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback]
        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_val_metrics = trainer.checkpoint_callback.best_model_score.item()
    results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    results[0]['best_val'] = best_val_metrics
    return results