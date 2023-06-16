# -*- coding: utf-8 -*-
"""
Ce fichier contient les fonctions nécessaires pour la modélisation
les données.
"""
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import eli5
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from eli5.sklearn import PermutationImportance


### ---------------------------------------------------Split Dataset----------------------------------------------------------###

##le notre
def split(database, alpha=.8):
    #Renvoie la base train et la base test
    lenght=database.shape[0]
    lenght1 =round(database.shape[0]*alpha)
    database_train = database[:lenght1].copy()
    database_test = database[lenght1:].copy()
    return database_train, database_test

##tuteur
def build_train_test_datasets(df_data,
                              input_cols=['Temperature', 'Humidity', 
                                          'WindSpeed', 'hour'], 
                              target_cols=["Consumption_Z1"],
                              train_ratio=0.8, scaler_str="Std"):
    """
    Permet de créer les ensembles d'apprentissage et de test en appliquant une
    normalisation si précisé en paramètres.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice des données d'entrée passée en paramètre.
    input_cols : list, optional
        La liste des colonnes considérées comme les entrées du modèle.
        La valeur par défaut est he default ['Temperature', 'Humidity', 
                                             'WindSpeed', 'hour'].
    target_col : str, optional
        Colonne considérée comme étant la valeur à prédire. 
        La valeur par défaut est "Consumption_Z1".
    train_ratio : float, optional
        La proportion des données devant être utilisé pour l'apprentissage.
        La valeur par défaut est 0.8.
    scaler_str : str, optional
        Un chaine de caractères indiquant quelle normalisation devrait être 
        appliquée aux données. Si la valur est valide, la normalisation est
        calculée sur les données d'apprentissage et appliquées sur l'ensemble
        des données. La valeur par défaut est 'Std' pour appliquer une
        normalisation centrée réduite.

    Returns
    -------
    d_train_test : dict
        Dictionnaire contenant les données d'apprentissage et de test
        normalisées et non normalisées.
    scaler : sklearn.preprocessing.MinMaxScaler
        Les coefficients de normalisation de l'opération "Min-Max" appliquée
        aux données d'apprentissage.

    """
    
    assert (train_ratio>0.6) & (train_ratio<=1),\
    "Le ratio de l\'ensemble d'apprentissage doit être dans l\'intervalle ]0.6, 1]"
    
    
    # Définir la base d'apprentissage en prenant un ratio de l'ensemble des données disponibles
    train_end_date = df_data.index[math.ceil(df_data.shape[0]*train_ratio)]
    train_inputs = df_data[input_cols].loc[:train_end_date]
    train_targets = df_data[target_cols].loc[:train_end_date]

    # La base de test est constituée du reste des données
    test_inputs = df_data[input_cols].loc[train_end_date:]
    test_targets = df_data[target_cols].loc[train_end_date:]
    
    d_train_test = dict()
    d_train_test["NonScaled"] = {"train":{"Inputs":train_inputs,
                                          "Target":train_targets},
                                 "test":{"Inputs":test_inputs,
                                         "Target":test_targets}}
    
    if scaler_str is None:
        scaler = None
        d_train_test["Scaled"] = None
    
    elif scaler_str.lower()=="minmax":
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd

        # Récupérer les indices temporelles des deux ensembles
        train_index = train_inputs.index
        test_index = test_inputs.index
        
        # Appliquer la normalisaton
        scaler = MinMaxScaler()
        train_inputs = scaler.fit_transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)

        # Convertir les valeurs normalisées en DataFrame
        train_inputs = pd.DataFrame(data=train_inputs, index=train_index,
                                    columns=input_cols)
        test_inputs = pd.DataFrame(data=test_inputs, index=test_index,
                                   columns=input_cols)
                                    
        d_train_test["Scaled"] = {"train":{"Inputs":train_inputs,
                                           "Target":train_targets},
                                 "test":{"Inputs":test_inputs,
                                         "Target":test_targets}}
        
    elif scaler_str.lower()=="std" or scaler_str.lower()=="standard":
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        # Récupérer les indices temporelles des deux ensembles
        train_index = train_inputs.index
        test_index = test_inputs.index
        
        # Appliquer la normalisaton
        scaler = StandardScaler()
        train_inputs = scaler.fit_transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)

        # Convertir les valeurs normalisées en DataFrame
        train_inputs = pd.DataFrame(data=train_inputs, index=train_index,
                                    columns=input_cols)
        test_inputs = pd.DataFrame(data=test_inputs, index=test_index,
                                   columns=input_cols)
                                    
        d_train_test["Scaled"] = {"train":{"Inputs":train_inputs, "Target":train_targets},
                                 "test":{"Inputs":test_inputs, "Target":test_targets}}
    else:
        scaler = None
        d_train_test["Scaled"] = None
        
    # Affichage du découpage de la base de données
    fig_ts, axis_ts = plt.subplots(1,1, figsize=(14, 8), sharey=True,
                             sharex=True, constrained_layout=True)  
    train_targets.plot(ax=axis_ts)
    test_targets.plot(ax=axis_ts)
    axis_ts.legend(["Données d'appreentissage", 'Données de test'])
    axis_ts.set_title("Découpage temporel données de test et d'apprentissage")

    return d_train_test, scaler

### ---------------------------------------------------Model RandomForest----------------------------------------------------------###

def simple_model(X_train,y_train,X_test,y_test,n_estimators=100):
    from sklearn.ensemble import RandomForestRegressor
    
    model=RandomForestRegressor(n_estimators=n_estimators,random_state=0, min_samples_split=5,
                                 min_samples_leaf=4, max_features='sqrt', max_depth=28,  bootstrap=True)
    model.fit(X_train,y_train,)
    prediction=model.predict(X_test)
    importances_values = model.feature_importances_
    importance = {}
    i = 0
    for column in X_test.columns:
        importance[column] = importances_values[i]
        i += 1
    print(f"The MAE of prediction for our model is:{ mean_absolute_error(prediction,y_test)}")
    return (model, importance)

#Feature importance 
# def feature_importance (X_test,y_test,model):
#     perm=PermutationImportance(model,random_state=1).fit(X_test,y_test)
#     eli5.show_weights(perm,feature_names = X_test.columns.tolist())
    
#
def detect_best_estimator(X_train,y_train,X_test,y_test,estimator):
    model=RandomForestRegressor(n_estimators=estimator,random_state=0)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    return mean_absolute_error(pred,y_test)



### ---------------------------------------------------Model Xgboost----------------------------------------------------------###

#notre
def model_performance(model, X_train, X_test, y_train, y_test):
    # Prédictions sur l'ensemble d'entraînement et de test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcul des mesures de performance
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Création du DataFrame
    performance = pd.DataFrame({
        'Score': ['R2', 'RMSE', 'MAE'],
        'Train': [r2_train, rmse_train, mae_train],
        'Test': [r2_test, rmse_test, mae_test]
    })

    return performance


#tuteur
def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=500,
                  b_feat_importance=True, b_verbose=True): 
    
    import xgboost as xgb 
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    # Définir le modèle
    model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                             n_estimators=n_estimators,
                             objective='reg:squarederror',
                             max_depth=10,
                             learning_rate=0.01, 
                             random_state=48)
    
    # Entraîner le modèle sur l'ensemble de test et l'évaluer également sur
    # l'ensemble de test
    if b_verbose:
        verbose = 100
    else:
        verbose = 0
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=verbose)
    
    # Afficher les performances du modèle
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    scores = dict()
    scores['R2'] = {"train": np.round(model.score(X_train, y_train), 3),
                    "test": np.round(model.score(X_test, y_test), 3)}
    scores['MAE'] = {"train": np.round(mean_absolute_error(y_train,
                                                           y_train_pred), 3),
                     "test": np.round(mean_absolute_error(y_test,
                                                          y_test_pred), 3)}

    print("Performance du modèle")
    print("\t --> Score R2 (Le modèle parfait a un score de 1)")
    print("\t \t --> Ensemble d'apprentissage : {}".format(scores['R2']['train']))
    print("\t \t --> Ensemble de test : {}".format(scores['R2']['test']))
    print("\n")
    print("\t --> MAE (Erreur moyenne absolue) en KW")
    print("\t \t --> Ensemble d'apprentissage : {}".format(scores['MAE']['train']))
    print("\t \t --> Ensemble de test : {}".format(scores['MAE']['test']))
    
    if b_feat_importance:
        import pandas as pd
        # Calculer l'importance des données d'entrée dans la prédiction
        feat_importances = pd.DataFrame(data=model.feature_importances_,
                                        index=X_train.columns,
                                        columns=['Feat_importance'])
        # Trier les poids des variables d'entrée afn d'avoir un affichage
        # explicite
        feat_importances = feat_importances.sort_values('Feat_importance')
        # Afficher le poids des différntes données d'entrée
        feat_importances.plot(kind='barh', title='Feature Importance')
        
    return model, scores
        

        
def apply_cross_validation(model, X_train, y_train, n_splits=10, n_repeats=3):
    
    assert n_splits >=2, "n_splits doit être supérieur ou égal à 2."
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import RepeatedKFold
    
    if model is None:
        import xgboost as xgb
        model = xgb.XGBRegressor(booster='gbtree', 
                                 objective='reg:squarederror')
    
    # define model evaluation method
    cross_val = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                              random_state=1)
    # evaluate model
    cv_results =\
        cross_validate(model, X_train, y_train, 
                       return_estimator=True, return_train_score=True,
                       scoring=('r2', 'neg_mean_absolute_error'), 
                       cv=cross_val, n_jobs=-1)
    
    return cv_results

### ------------------------------------------------------Model LSTM----------------------------------------------------------###
# def create_uncompiled_model():
#   # define a sequential model
#   model = tf.keras.models.Sequential([ 
#       tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                     input_shape=[None]),
#       tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
#       tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
#       tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
#       tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
#       tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#       tf.keras.layers.Dense(1),
#   ]) 

#   return model


# class EarlyStopping(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
    
#     if(logs.get('mae') < 0.03):
#       print("\nMAEthreshold reached. Training stopped.")
#       self.model.stop_training = True

# # Let's create an object of our class and assign it to a variable
# early_stopping = EarlyStopping()


