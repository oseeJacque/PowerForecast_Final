#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np

"""
Ce fichier regroupe quelques fonctions utiles pour analyser les données.
"""

from datetime import datetime

def apply_sanity_check(df_dataset):

    assert isinstance(df_dataset.index, pd.DatetimeIndex), \
    "Les indices de la matrice de données doivent être le temps"

    d_sanity_check = dict()
    # Compter les valeurs invalides de la base de données
    d_sanity_check["Valeurs_invalides"] = dict(df_dataset.isnull().sum())

    # Vérifier la fréquence d'échantillonnage
    time_diff = df_dataset.index.to_series().diff().astype('timedelta64[s]').values

    # Supprimer la première valeur parce que la différence pour le premier pas de temps
    time_diff = np.sum(np.diff(time_diff[1:])).astype(float)
       
    if time_diff < 10e-3:
        d_sanity_check['Periode_echantillonnage'] = "Uniforme"
    else:
        d_sanity_check['Periode_echantillonnage'] = "Non Uniform"
  
    return d_sanity_check


def plot_corr(df, figsize=(10, 10)):
    """
    Permet d'afficher la matrice de corrélation de la matrice des données
    passée en paramètres.

    Parameters
    ----------
    df : pandas.DataFrame
        Matrice de données d'entrée.
    figsize : tuple, optional
        Dimensions de la figure utilisée pour l'affichage de la matrice de
        corrélation. La valeur par défaut est (10, 10).

    Returns
    -------
    None.

    """
 

    # Calcul de la matrice de corrélation
    corr = df.corr()

    # Créer une figure pour la matrice de corrélation
    fig, ax = plt.subplots(figsize=figsize)

    # Générer une palette de couleur personnalisée
    cmap = sns.diverging_palette(210, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, center=0,square=True,
                linewidths=.5, cbar_kws={"shrink": .5})

    
def date_to_number(year, month,day):
    format_date = date(year,month,day).isoformat()
    return df_energy.loc[format_dat]['DayOfYear'].unique()[0]

 
def display_consumption_monthly(df_data, figsize=(20, 15), month_num="all"):
    """
    Permet d'afficher les séries temporelles des consommations mensuelles
    des 3 zones.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice des données passée en paramètres.
    figsize : tuple, optional
        La taille de la fenêtre utilisée pour l'affichage. La valeur par 
        défaut est (20, 15).
    month_num : int, optional
        Le numéro de mois concerné par l'affichage. Cette valeur doit être
        posiive et inférieure ou égale à 12. Pour afficher toutes
        les données utilisées on peut indiquer "all". 
        La valeur par défaut est "all".

    Raises
    ------
    ValueError
        Exception levée quand la valeur du numéro du mois n'est pas valide.

    Returns
    -------
    None.

    """

    assert "Month" in df_data.columns, \
        "La colonne \'Month' n\'est pas présente dans la matrice de données"
    
    if isinstance(month_num, str):
        if month_num.lower()=="all":
            df_sub_data = df_data.copy()
        else:
            df_sub_data = None
            print("Le numéro du mois de l'année n'est pas valide : "
                  "valeur positive et inférieure à 12.")
            raise ValueError
    elif isinstance(month_num, int):
        if (month_num <= 12) & (month_num > 0):
            df_sub_data = df_data[df_data["Month"]==month_num]
        else:
            df_sub_data = None
            print("Le numéro du mois de l'année n'est pas valide : "
                  "valeur positive et inférieure à 12.")
            raise ValueError
    else:
            df_sub_data = None
            print("Le numéro du mois de l'année n'est pas valide : "
                  "valeur positive et inférieure à 12.")
            raise ValueError
    
    if df_sub_data is not None:
        
            
        mask_zone = ['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']
        fig, ax = plt.subplots(len(mask_zone), 1, figsize=(15, 7), sharey=True, tight_layout=True)
        #ax_multiplot = df_sub_data[mask_zone].plot(subplots=True, sharey=True, figsize=(15, 7))
        #ax_multiplot[0].set_ylabel("Power consumption (KW)")
        #ax_multiplot[1].set_ylabel("Power consumption (KW)")
        #ax_multiplot[2].set_ylabel("Power consumption (KW)")
        for i, zone in enumerate(mask_zone):
            ax[i].plot(df_sub_data['DateTime'], df_sub_data[zone])
            ax[i].set_ylabel("Power consumption (KW)")
        st.pyplot(fig)
    
def display_consumption_daily(df_data, figsize=(25, 12), day_num = "all"):
    """
    Permet d'afficher les séries temporelles relatives au consommations
    journalières.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La base de données passée en paramètres.
    figsize : tuple, optional
        Dimensions de la figure servant à l'affichage des consommations
        journalières. La valeur par défaut est (20, 15).
    day_num : int, optional
        Numéro du jour de l'année concerné par l'affichage. Cette valeur doit
        être positive et inférieure ou égale à 365. 
        La valeur par défaut est "all".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    assert "DayOfYear" in df_data.columns,\
        "La colonne \'DayOfYear' n\'est pas présente dans la matrice de données"
    
    if isinstance(day_num, str):
        if day_num.lower()=="all":
            df_sub_data = df_data.copy()
        else:
            df_sub_data = None
            print("Le numéro du jour de l'année n'est pas valide : "
                  "valeur positive et inférieure à 365.")
            raise ValueError
    elif isinstance(day_num, int):
        if (day_num <= 365) & (day_num > 0):
            df_sub_data = df_data[df_data["DayOfYear"]==day_num]
        else:
            df_sub_data = None
            print("Le numéro du jour de l'année n'est pas valide : "
                  "valeur positive et inférieure à 365.")
            raise ValueError
    else:
            df_sub_data = None
            print("Le numéro du jour de l'année n'est pas valide : "
                  "valeur positive et inférieure à 365.")
            raise ValueError
    
    if df_sub_data is not None:
        mask_zone = ['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']
        fig, ax = plt.subplots(len(mask_zone), 1, figsize=figsize, sharey=True, tight_layout=True)

        for i, zone in enumerate(mask_zone):
            ax[i].plot(df_sub_data['DateTime'], df_sub_data[zone])
            ax[i].set_ylabel("Power consumption (KW)")

        st.pyplot(fig)
        
   
def display_consumption_period(df_data, figsize=(20, 15), start_date="2017-01-01" , end_date="2017-12-30"):
    
    """
    Permet d'afficher les séries temporelles des consommations mensuelles
    des 3 zones sur une période de temps.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice des données passée en paramètres.
    figsize : tuple, optional
        La taille de la fenêtre utilisée pour l'affichage. La valeur par 
        défaut est (20, 15).
    start_date : datetime, optional
        La date de début de la période concerné par l'affichage. Cette date 
        doit être comprise dans l'année 2017
    end_date : datetime, optional
        La date de fin de la période concerné par l'affichage. Cette date 
        doit être comprise dans l'année 2017 et supérieure à la date de début.

    Raises
    ------
    ValueError
        Exception levée quand les dates ne sont pas dans l'année 2017 et également 
        lorsque la date de fin est inférieur à la date de début.

    Returns
    -------
    None.

    """
    
    
    assert "start_date" or "end_date" in df_data.index, \
         "La colonne \'Month' n\'est pas présente dans la matrice de données"
    yearofstudy = pd.date_range(start='2017-01-01', end='2017-12-31')
    if isinstance(start_date, str):
        if start_date in yearofstudy and end_date in yearofstudy:
            if datetime.strptime(end_date, '%Y-%m-%d') >= datetime.strptime(start_date, '%Y-%m-%d'):
                df_sub_data = df_data.loc[(df_data.DateTime >= start_date) & (df_data.DateTime <= end_date)]
    else:
        df_sub_data = None
        print("La période n'est pas valide : "
              "définissez une période compris dans l'année 2017.")
        raise ValueError
    
    if df_sub_data is not None:
        mask_zone = ['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']
        fig, ax = plt.subplots(len(mask_zone), 1, figsize=figsize, sharey=True, tight_layout=True)

        for i, zone in enumerate(mask_zone):
            ax[i].plot(df_sub_data['DateTime'], df_sub_data[zone])
            ax[i].set_ylabel("Power consumption (KW)")

        st.pyplot(fig)
    

        
def display_consumption_period2(df_data, figsize=(20, 15), start_date="2017-01-01", end_date="2017-12-30"):
    """
    Permet d'afficher les séries temporelles des consommations mensuelles
    des 3 zones sur une période de temps.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice des données passée en paramètres.
    figsize : tuple, optional
        La taille de la fenêtre utilisée pour l'affichage. La valeur par 
        défaut est (20, 15).
    start_date : datetime, optional
        La date de début de la période concernée par l'affichage. Cette date 
        doit être comprise dans l'année 2017.
    end_date : datetime, optional
        La date de fin de la période concernée par l'affichage. Cette date 
        doit être comprise dans l'année 2017 et supérieure à la date de début.

    Raises
    ------
    ValueError
        Exception levée quand les dates ne sont pas dans l'année 2017 et également 
        lorsque la date de fin est inférieure à la date de début.

    Returns
    -------
    None.
    """
    
    assert "start_date" or "end_date" in df_data.index, \
        "La colonne 'Month' n'est pas présente dans la matrice de données"
    
    year_of_study = pd.date_range(start='2017-01-01', end='2017-12-31')
    
    if isinstance(start_date, str): 
        if start_date in year_of_study and end_date in year_of_study:
            if pd.to_datetime(end_date) >= pd.to_datetime(start_date):
                df_sub_data = df_data.loc[start_date:end_date]
            else:
                raise ValueError("La date de fin doit être supérieure à la date de début.")
        else:
            raise ValueError("Les dates doivent être comprises dans l'année 2017.")
    else:
        raise ValueError("Veuillez spécifier les dates au format 'YYYY-MM-DD'.")
    
    if df_sub_data is not None:
        mask_zone = ['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']
        
        fig, axes = plt.subplots(nrows=3, figsize=figsize)
        for i, zone in enumerate(mask_zone):
            df_sub_data.plot(y=zone, ax=axes[i])
            axes[i].set_ylabel("Power consumption (KW)")
        
        st.pyplot(fig)
                
def display_consumption_resume(df_data, scale="Hours", figsize=(10, 8)):
    """
    Permet d'afficher les consommations de façon compacte à l'aide de boite à
    moustaches sur une echelle mensuelle ou journalière.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice contenant les données d'entrée passées en paramètres.
    scale : str, optional
        L'achelle de temps sur laquelle on veut analyser les enregistrements.
        Seules les échelles journalière (Hours) et mensuelle (Months) 
        sont disponibles. La valeur par défaut est "Hours".
    figsize : tuple, optional
        Dimensions de la figure utilisée pour l'affichage. La valeur par 
        défaut est (10, 8).

    Raises
    ------
    ValueError
        DESCRIPTION.
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if isinstance(scale, str):
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        if scale.lower()=="hours":
            fig, axis = plt.subplots(3,1, figsize=figsize, sharey=True,
                                     sharex=True, constrained_layout=True)    
        
            sns.boxplot(data=df_data, x='Hour', y='Zone 1 Power Consumption', 
                        ax=axis[0], palette='Reds')
            sns.boxplot(data=df_data, x='Hour', y='Zone 2 Power Consumption', 
                        ax=axis[1], palette='Blues')
            sns.boxplot(data=df_data, x='Hour', y='Zone 3 Power Consumption', 
                        ax=axis[2], palette='Greens')
            axis[0].set_title('Zone 1 consumption power (KW) by Hour')
            axis[1].set_title('Zone 2 consumption power (KW) by Hour')
            axis[2].set_title('Zone 3 consumption power (KW) by Hour')
            
            for i in range(3):
                axis[i].set_xlabel('Hours')
                axis[i].set_ylabel('Consumption power (KW)')
        elif scale.lower()=="months":
            fig, axis = plt.subplots(3,1, figsize=figsize, sharey=True,
                                     sharex=True, constrained_layout=True) 
            sns.boxplot(data=df_data, x='Month', y='Zone 1 Power Consumption', 
                        ax=axis[0], palette='Reds')
            sns.boxplot(data=df_data, x='Month', y='Zone 2 Power Consumption',
                        ax=axis[1], palette='Blues')
            sns.boxplot(data=df_data, x='Month', y='Zone 3 Power Consumption', 
                        ax=axis[2], palette='Greens')
            axis[0].set_title('Zone 1 consumption power (KW) by Month')
            axis[1].set_title('Zone 2 consumption power (KW) by Month')
            axis[2].set_title('Zone 3 consumption power (KW) by Month')
            
            for i in range(3):
                axis[i].set_xlabel('Months')
                axis[i].set_ylabel('Consumption power (KW)')
        else:
            print("La valeur de l'échelle de temps n'est pas conforme: '"
                  "Hours' ou 'Months'.")
            raise ValueError
    else:
        print("L\'échelle de temps doit être une chaine de caractères.")
        raise TypeError
              

