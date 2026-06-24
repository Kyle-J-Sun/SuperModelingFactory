from __init__ import *

def save_model(model, filename):
    """ Save lightGBM model using pickle. """
    import joblib
    
    joblib.dump(model, filename)
    return 0

def load_model(model_path):
    """ Load Pickle Model. """
    import joblib
    
    model = joblib.load(model_path)
    return model

def scoring(data, model, varlist, scr_name, keeplist = None, all_missing_spec_value = None):
    """ Model Soring. """
    
    fnl_data = data.copy()
    fnl_data[scr_name] = model.predict_proba(fnl_data.loc[:, varlist])[:, 1]
    
    nohit_condition = (pd.isnull(fnl_data[varlist]).sum(axis = 1) == len(varlist))
    if fnl_data[nohit_condition].shape[0] > 0:
        
        all_missing_data = fnl_data[nohit_condition]
        other_data = fnl_data[~nohit_condition]
        
        all_missing_data[scr_name] = model.predict_proba(fnl_data[nohit_condition].loc[:, varlist])[:, 1]
        print("Score for All-Missing Cases: ", all_missing_data[scr_name].unique())
        
        if all_missing_spec_value:
            all_missing_data[scr_name] = all_missing_spec_value
            print(f"Score for All-Missing Cases Has Been Reset to {all_missing_spec_value}")
        
        fnl_data = pd.concat([other_data, all_missing_data])
        
    assert fnl_data.shape[0] == data.shape[0]
    
    if keeplist is None:
        keeplist = fnl_data.columns.tolist()
    else:
        keeplist = keeplist + [scr_name]
    
    return fnl_data[keeplist]

def upload_score(data, model, varlist, scr_name, table_name, keeplist = None, retPandas = False, all_missing_spec_value = None):
    """ Upload Score to Maxcompute. """
    
    data = scoring(data = data, model = model, varlist = varlist, scr_name = scr_name, keeplist = keeplist, all_missing_spec_value = all_missing_spec_value)
    
    sqlrunner = ODPSRunner()

    fnl_scr_upload = data.copy()
    fnl_scr_upload = uf.npnan2none(fnl_scr_upload)
    fnl_scr_upload = uf.drop_tmp_cols(fnl_scr_upload)
    
    if keeplist is None:
        keeplist = fnl_scr_upload.columns.tolist()
    else:
        keeplist = keeplist + [scr_name]
    
    sqlrunner.upload_df(fnl_scr_upload[keeplist], table_name)
    
    if retPandas:
        return fnl_scr_upload[keeplist]
    
    return 0