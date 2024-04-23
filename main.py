import datetime
import sys
import warnings
import numpy as np

import torch
from torch_geometric.nn.models import MetaPath2Vec

from hgmae.models.edcoder import PreModel
from hgmae.utils import (evaluate, evaluate_cluster, load_best_configs, load_data,
                         metapath2vec_train, preprocess_features,
                         set_random_seed)
from hgmae.utils.params import build_args

warnings.filterwarnings('ignore')


def main(args):
    # random seed
    set_random_seed(args.seed)

    # load data
    (nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test), g, processed_metapaths = \
        load_data(args.dataset, args.ratio, args.type_num)
    # nei_index: a list of torch.tensor, each tensor is the neighbor index of a node len(nei_index[0]) = len(feats[0])
    # feats: list of torch.tensor, len(feats) = 2; len(feats[0]) = len(feats[1]) = 4057
    # mps: a list of torch.tensor, len(mps) = 3; mps[0].coalesce().indices().shape = (2, 11113);
    #      mps[1].coalesce().indices().shape = (2, 5000495); mps[2].coalesce().indices().shape = (2, 7043571)
    # pos: torch.tensor, pos.coalesce().indices().shape = (2, 4046685)
    # label: torch.tensor, label.shape = (4057, 4)
    # idx_train: list of torch.tens  or, len(idx_train) = 3; len(idx_train[0]) = 80; len(idx_train[1]) = 160; len(idx_train[2]) = 240
    # idx_val: list of torch.tensor, len(idx_val) = 3; len(idx_val[0]) = 1000; len(idx_val[1]) = 1000; len(idx_val[2]) = 1000
    # idx_test: list of torch.tensor, len(idx_test) = 3; len(idx_test[0]) = 1000; len(idx_test[1]) = 1000; len(idx_test[2]) = 1000
    # g: HeteroData(
    #   (target, relation_0-->, dst_0)={ edge_index=[2, 19645] },
    #   (dst_0, <--relation_0, target)={ edge_index=[2, 19645] }
    # )
    # processed_metapaths: [('target', 'relation_0-->', 'dst_0'), ('dst_0', '<--relation_0', 'target')]
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]


    num_mp = int(len(mps))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", num_mp)

    if args.use_mp2vec_feat_pred:
        # metapath2vec was used in default setting
        assert args.mps_embedding_dim > 0
        metapath_model = MetaPath2Vec(g.edge_index_dict,
                                      args.mps_embedding_dim,# 64
                                      processed_metapaths,# [('target', 'relation_0-->', 'dst_0'), ('dst_0', '<--relation_0', 'target')]
                                      args.mps_walk_length,# 5
                                      args.mps_context_size,# 3
                                      args.mps_walks_per_node,# 3
                                      args.mps_num_negative_samples,# 1
                                      sparse=True
                                      )
        metapath2vec_train(args, metapath_model, args.mps_epoch, args.device)
        mp2vec_feat = metapath_model('target').detach()
        # mp2vec_feat: torch.tensor, mp2vec_feat.shape = (4057, 64)
        # free up memory
        del metapath_model
        if args.device.type == 'cuda':
            mp2vec_feat = mp2vec_feat.cpu()
            torch.cuda.empty_cache()
        mp2vec_feat = torch.FloatTensor(preprocess_features(mp2vec_feat))
        feats[0] = torch.hstack([feats[0], mp2vec_feat])

    # model
    focused_feature_dim = feats_dim_list[0]
    model = PreModel(args, num_mp, focused_feature_dim)
    # debuger set trace
    # import pdb; pdb.set_trace()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    # scheduler
    if args.scheduler:
        print("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    model.to(args.device)# move model to device
    feats = [feat.to(args.device) for feat in feats] # move feats to device, most likely cuda
    mps = [mp.to(args.device) for mp in mps]
    label = label.to(args.device)
    idx_train = [i.to(args.device) for i in idx_train]
    idx_val = [i.to(args.device) for i in idx_val]
    idx_test = [i.to(args.device) for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    best_model_state_dict = None
    for epoch in range(args.mae_epochs):
        model.train()
        optimizer.zero_grad()
        loss, loss_item = model(feats, mps, nei_index=nei_index, epoch=epoch)
        print(f"Epoch: {epoch}, loss: {loss_item}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            best_model_state_dict = model.state_dict()
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    print('The best epoch is: ', best_t)
    model.load_state_dict(best_model_state_dict) # load the best model state dict
    model.eval() # set model to evaluation mode
    embeds = model.get_embeds(feats, mps, nei_index) # embeds: torch.tensor, embeds.shape = (4057, 256)
    # debuger set trace
    import pdb; pdb.set_trace()
    if args.task == 'classification':
        macro_score_list, micro_score_list, auc_score_list = [], [], []
        for i in range(len(idx_train)):
            macro_score, micro_score, auc_score = evaluate(embeds, idx_train[i], idx_val[i], idx_test[i], label, nb_classes, args.device,
                                                           args.eva_lr, args.eva_wd)
            macro_score_list.append(macro_score)
            micro_score_list.append(micro_score)
            auc_score_list.append(auc_score)
    elif args.task == 'clustering':
        # node clustering
        nmi_list, ari_list = [], []
        embeds = embeds.cpu().data.numpy()
        label = np.argmax(label.cpu().data.numpy(), axis=-1)
        for kmeans_random_state in range(10):
            nmi, ari = evaluate_cluster(embeds, label, args.n_labels, kmeans_random_state)
            nmi_list.append(nmi)
            ari_list.append(ari)
        print("\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]".format(np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)))
    else:
        sys.exit('wrong args.task.')

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")


if __name__ == "__main__":
    args = build_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        args.device = torch.device("cpu")

    if args.use_cfg:
        if args.task == 'classification':
            config_file_name = "configs.yml"
        elif args.task == 'clustering':
            config_file_name = "clustering_configs.yml"
        else:
            sys.exit(f'No available config file for task: {args.task}')
        args = load_best_configs(args, config_file_name)
    print(args)

    main(args)
