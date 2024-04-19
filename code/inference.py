from os.path import join
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
from utils.modules import MM_TN, MM_TNConfig
from utils.datautils import SMILESProteinDataset
from utils.train_utils import *


BASE_DIR = join(".")


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


MODELS_DIR = join(BASE_DIR, "data", "saved_models")

pretrained_TN = join(MODELS_DIR, "ProSmith", "ESP_new_smiles_2gpus_bs48_1e-05_layers6.txt.pkl")

xgb_ESM1b_ChemBERTa = join(MODELS_DIR, "xgboost", "ESP", "ESM1b_ChemBERTa.pkl")
xgb_ESM1b_ChemBERTa_cls = join(MODELS_DIR,"xgboost", "ESP", "ESM1b_ChemBERTa_cls.pkl")
xgb_cls = join(MODELS_DIR, "xgboost", "ESP", "cls.pkl")


def extract_repr(model, dataloader):
    # evaluate the model on validation set
    model.eval()

    if is_cuda(device):
        model = model.to(device)
    #print("Extracting representation for ProSmith")
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            
            # move batch to device
            batch = [r.to(device) for r in batch]
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels, indices = batch
            if is_cuda(device):
                labels = labels.to(device)
                indices = indices.to(device)
                smiles_emb = smiles_emb.to(device)
                smiles_attn = smiles_attn.to(device)
                protein_emb = protein_emb.to(device)
                protein_attn = protein_attn.to(device)

            _, cls_repr = model(smiles_emb=smiles_emb,
                                smiles_attn=smiles_attn, 
                                protein_emb=protein_emb,
                                protein_attn=protein_attn,
                                device=device,
                                gpu = 0,
                                get_repr=True)

            protein_attn = int(sum(protein_attn.cpu().detach().numpy()[0]))
            smiles_attn = int(sum(smiles_attn.cpu().detach().numpy()[0]))

            
                
            smiles = smiles_emb[0][:smiles_attn].mean(0).cpu().detach().numpy()
            esm1b = protein_emb[0][:protein_attn].mean(0).cpu().detach().numpy()
            cls_rep = cls_repr[0].cpu().detach().numpy()

            if step ==0:
                cls_repr_all = cls_rep.reshape(1,-1)
                esm1b_repr_all = esm1b.reshape(1,-1)
                smiles_repr_all = smiles.reshape(1,-1)
                labels_all = labels[0]
                #print(indices.cpu().detach().numpy())
                orginal_indices = list(indices.cpu().detach().numpy())


            else:
                cls_repr_all = np.concatenate((cls_repr_all, cls_rep.reshape(1,-1)), axis=0)
                smiles_repr_all = np.concatenate((smiles_repr_all, smiles.reshape(1,-1)), axis=0)
                esm1b_repr_all = np.concatenate((esm1b_repr_all, esm1b.reshape(1,-1)), axis=0)
                labels_all = torch.cat((labels_all, labels[0]), dim=0)
                orginal_indices = orginal_indices + list(indices.cpu().detach().numpy())

    return cls_repr_all, esm1b_repr_all, smiles_repr_all, labels_all.cpu().detach().numpy(), orginal_indices




def inference(df):
    embedding_path = join(BASE_DIR, "data", "temp_embeddings")

    config = MM_TNConfig.from_dict({"s_hidden_size": 600,
        "p_hidden_size":1280,
        "hidden_size": 768,
        "max_seq_len": 1276,
        "num_hidden_layers" : 6,
        "binary_task" : True})

    #print(f"Loading dataset")

    test_dataset = SMILESProteinDataset(df=df,
                                        embed_dir = embedding_path,
                                        train=False, 
                                        device=device, 
                                        gpu=0,
                                        random_state = 0,
                                        binary_task = True,
                                        extraction_mode = True)

    #print(f"Loading dataloader")
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #print(f"Loading model")
    model = MM_TN(config)
    
    if is_cuda(device):
        model = model.to(device)

    #Loading weights from pretrained model
    #(f"Loading model")
    state_dict = torch.load(pretrained_TN, map_location=torch.device(device))



 
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("module.", "")] = value
    model.load_state_dict(new_state_dict)
    #print("Successfully loaded pretrained model")


    test_cls, test_esm1b, test_smiles, test_labels, test_indices = extract_repr(model, testloader)
    #print(f"Extraction complete")
    
    def get_predictions(dM_val, save_name):
        with open(save_name, 'rb') as file:
            bst = pickle.load(file)
        y_val_pred = bst.predict(dM_val)
        return(y_val_pred)

    

    ############# ESM1b+ChemBERTa2
    test_X_all = np.concatenate([test_esm1b, test_smiles], axis = 1)
    dtest = xgb.DMatrix(np.array(test_X_all), label = np.array(test_labels).astype(float))

    #print("ESM1b+ChemBERTa2")
    y_test_pred_all = get_predictions(dM_val = dtest, save_name = xgb_ESM1b_ChemBERTa)


    ############# ESM1b+ChemBERTa +cls
    test_X_all_cls = np.concatenate([np.concatenate([test_esm1b, test_smiles], axis = 1), test_cls], axis=1)
    dtest_all_cls = xgb.DMatrix(np.array(test_X_all_cls), label = np.array(test_labels).astype(float))

    #print("ESM1b+ChemBERTa2+cls-token")
    y_test_pred_all_cls = get_predictions(dM_val = dtest_all_cls, save_name = xgb_ESM1b_ChemBERTa_cls)


    ############# cls token
    dtest_cls = xgb.DMatrix(np.array(test_cls), label = np.array(test_labels).astype(float))         
    #print("cls-token")
    y_test_pred_cls = get_predictions(dM_val = dtest_cls, save_name = xgb_cls)

        
    #### Final predictions:
    best_i, best_j, best_k = 0.45, 0.50, 0.05
    y_test_pred = best_i*y_test_pred_all_cls + best_j*y_test_pred_all + best_k*y_test_pred_cls
    #print("Three models combined:")
    #print("ESM1b+ChemBERTa2+cls: %s, ESM1b+ChemBERTa2: %s, cls-token: %s" %(best_i, best_j, best_k))


    df = df.reset_index(drop=True)

    #y_test pred has the predictions for df in the order of test_indices
    y_test_pred = y_test_pred.flatten()  # Ensure y_test_pred is a flat array
    y_test_pred = y_test_pred[test_indices]  # Assuming test_indices is correctly defined
    df["predictions"] = y_test_pred
    return(df)