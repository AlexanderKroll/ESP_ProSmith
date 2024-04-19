import shutil
import os
from os.path import join
import pandas as pd
from smiles_embeddings import *
from protein_embeddings import *
from util_embeddings import convert_to_smiles, validate_enzyme
from inference import inference

BASE_DIR = join(".")

def ESP_predicton(metabolite_list, enzyme_list):
    try:
        #make new directory:
        try:
            os.mkdir(join(BASE_DIR, "data", "temp_embeddings"))
        except:
            shutil.rmtree(join(BASE_DIR, "data", "temp_embeddings"))
            os.mkdir(join(BASE_DIR, "data", "temp_embeddings"))
            
        df_ESP = pd.DataFrame({"metabolite" : metabolite_list, "enzyme" : enzyme_list,
                                "index" : list(range(len(metabolite_list)))})
        df_ESP["valid"] = True
                    
        print("Step 1/2: Preprocessing: Calculating input embeddings.")
        df_ESP = preprocessing(df = df_ESP)
        df_ESP_valid, df_ESP_invalid = df_ESP.loc[df_ESP["valid"]], df_ESP.loc[~df_ESP["valid"]]
        df_ESP_valid.reset_index(inplace = True, drop = True)

        print("Step 2/2: Making predictions for ESP.")
        df_ESP_valid = inference(df = df_ESP_valid)

        df_ESP = pd.concat([df_ESP_valid, df_ESP_invalid], ignore_index = True)
        df_ESP = df_ESP.sort_values(by = ["index"])
        df_ESP.drop(columns = ["index"], inplace = True)
        df_ESP.reset_index(inplace = True, drop = True)
        df_ESP = process_df_columns(df_ESP)

        #remove temporary embeddings directory:
        shutil.rmtree(join(BASE_DIR, "data", "temp_embeddings"))
        return(df_ESP)
    except Exception as e:
        error_message = str(e)
        print("Error:" + error_message)
        return(None)
    

def process_df_columns(df):
    df = df.drop(columns = ["SMILES"])
    df = df.rename(columns = {"valid" : "valid input", "predictions" : "Prediction score",
                              "metabolite" : "Metabolite", "enzyme" : "Protein"})
    df["Prediction score"] = df["Prediction score"].apply(lambda x: round(x, 2))
    return(df)


def preprocessing(df):
    #convert all metabolites to SMILES strings:
    df["SMILES"] = df["metabolite"].apply(lambda x: convert_to_smiles(x))
    df["valid"].loc[pd.isnull(df["SMILES"])] = False

    #check if enzyme is valid:
    for ind in df.index:
        if not validate_enzyme(df["enzyme"][ind]):
            df["valid"][ind] = False

    #Get all Protein Sequences and SMILES strings
    all_sequences = list(set(df["enzyme"].loc[df["valid"] == True]))
    all_smiles = list(set(df["SMILES"].loc[df["valid"] == True]))
   
   
    #Calculate embeddings:
    print(".....1(a) Calculating protein embeddings")
    calculate_protein_embeddings(all_sequences, 1000)
    print(".....1(b) Calculating SMILES embeddings")
    calculate_smiles_embeddings(all_smiles, 1000)
    return(df)
    