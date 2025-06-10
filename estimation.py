import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, norm, t, chi2, probplot
import plotly.express as px
from statsmodels.stats.proportion import proportion_confint
# from scipy.stats import binom # Remplacé par statsmodels qui est plus complet

# Configuration de la page
st.set_page_config(layout="wide", page_title="📊 Analyse Statistique")

# --- Widget déplacé en dehors de la fonction mise en cache ---
uploaded_file = st.sidebar.file_uploader(
    "📤 Importer un fichier (CSV/Excel)",
    type=['csv', 'xlsx']
)

# Fonction pour charger les données (prend maintenant uploaded_file en argument)
@st.cache_data
def load_data(uploaded_file_obj):
    if uploaded_file_obj is not None:
        try:
            # Utiliser un nouveau nom de variable pour éviter l'écrasement
            file_name_ext = uploaded_file_obj.name
            if file_name_ext.endswith('.csv'):
                df_loaded = pd.read_csv(uploaded_file_obj) # Utiliser un nom différent
            else:
                df_loaded = pd.read_excel(uploaded_file_obj) # Utiliser un nom différent
            return df_loaded
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
            return None
    return None

# Fonction bootstrap (inchangée)
def bootstrap_ci(data, func, n_resamples=1000, conf_level=0.95):
    # S'assurer que les données sont un array numpy pour le bootstrap
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    # Gérer le cas où les données sont vides ou ont moins de 2 éléments pour var/std
    if len(data) < 1 or (func in [np.var, np.std] and len(data) < 2) :
         return (np.nan, np.nan)
    resamples = np.random.choice(data, size=(n_resamples, len(data)), replace=True)
    # Appliquer la fonction en gérant les lignes avec moins de 2 éléments pour var/std si nécessaire
    if func in [np.var, np.std] :
         estimates = np.array([func(resample, ddof=1) if len(resample) >= 2 else np.nan for resample in resamples])
         estimates = estimates[~np.isnan(estimates)] # Exclure les NaN
         if len(estimates) < 2 : return (np.nan, np.nan) # Pas assez de données valides pour l'intervalle
    else :
        estimates = np.apply_along_axis(func, 1, resamples)

    alpha = 1 - conf_level
    # Gérer le cas où il n'y a pas assez d'estimations valides
    if len(estimates) == 0:
        return (np.nan, np.nan)
    return np.percentile(estimates, [100*alpha/2, 100*(1-alpha/2)])


# --- Flux d'exécution principal ---

# Charger les données uniquement si un fichier est téléversé
if uploaded_file is not None:
    df = load_data(uploaded_file) # df est maintenant le dataframe chargé

    # Continuer seulement si le chargement des données a réussi
    if df is not None:
        st.sidebar.write(f"🔍 {len(df)} observations, {len(df.columns)} variables")

        # Onglets principaux
        tab1, tab2 = st.tabs(["Variables Quantitatives", "Variables Qualitatives"])

        with tab1:
            # Partie Variables Quantitatives
            # Sélection de la variable quantitative
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                 # Ce header s'affiche seulement s'il n'y a pas de var. quantitatives
                st.header("Analyse des Variables Quantitatives")
                st.warning("Aucune variable quantitative trouvée dans les données.")
            else:
                selected_quant_col = st.selectbox("Sélectionnez la variable quantitative", numeric_cols, key="quant_var_select")

                # --- Header dynamique ---
                st.header(f"Analyse de la Variable Quantitative : `{selected_quant_col}`")

                # Nettoyage des données pour la variable sélectionnée
                # Utiliser .copy() pour éviter SettingWithCopyWarning si on modifie clean_quant_data plus tard
                clean_quant_data = df[selected_quant_col].dropna().copy()

                if len(clean_quant_data) > 0:
                    # Affichage des statistiques descriptives
                    # --- Sous-titre dynamique ---
                    st.subheader(f"Statistiques descriptives pour `{selected_quant_col}`")
                    col1_desc, col2_desc = st.columns(2) # Noms de colonnes uniques

                    with col1_desc:
                        stats_dict = {
                            "Moyenne": clean_quant_data.mean(),
                            "Médiane": clean_quant_data.median(),
                            "Écart-type (Éch.)": clean_quant_data.std(ddof=1), # Préciser Échantillon
                            "Variance (Éch.)": clean_quant_data.var(ddof=1), # Préciser Échantillon
                            "Minimum": clean_quant_data.min(),
                            "Maximum": clean_quant_data.max(),
                            "Effectif (non NA)": len(clean_quant_data) # Plus précis
                        }
                        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Valeur'])
                        st.dataframe(stats_df.style.format("{:.4f}"))

                    with col2_desc:
                        fig_box, ax_box = plt.subplots() # Noms de fig/ax uniques
                        sns.boxplot(x=clean_quant_data, ax=ax_box, color='skyblue')
                        # --- Label dynamique déjà présent ---
                        ax_box.set_xlabel(selected_quant_col)
                        ax_box.set_title(f'Box Plot de `{selected_quant_col}`') # Titre dynamique
                        st.pyplot(fig_box)

                    # Test de normalité
                    # --- Sous-titre dynamique ---
                    st.subheader(f"Test de normalité pour `{selected_quant_col}`")
                    test_type = st.radio("Type de test", ["Shapiro-Wilk", "Kolmogorov-Smirnov"],
                                         key=f"norm_test_type_{selected_quant_col}", horizontal=True) # Clé unique par variable

                    # Effectuer le test seulement si assez de données (>3 pour Shapiro)
                    p_value = np.nan # Initialiser
                    stat = np.nan
                    normality_test_done = False
                    if test_type == "Shapiro-Wilk" and len(clean_quant_data) >= 3:
                        stat, p_value = shapiro(clean_quant_data)
                        normality_test_done = True
                    elif test_type == "Kolmogorov-Smirnov" and len(clean_quant_data) > 0:
                         # K-S test contre une distribution normale avec moyenne/écart-type estimés
                        stat, p_value = kstest(clean_quant_data, 'norm',
                                               args=(clean_quant_data.mean(), clean_quant_data.std(ddof=1))) # Utiliser std échantillon
                        normality_test_done = True

                    if normality_test_done:
                        st.write(f"**Test:** {test_type}")
                        st.write(f"**Statistique:** {stat:.4f}")
                        st.write(f"**P-value:** {p_value:.4f}")
                        alpha_norm = 0.05 # Seuil alpha
                        if p_value > alpha_norm:
                             conclusion = f"✅ Ne pas rejeter H0. La distribution de `{selected_quant_col}` pourrait être normale (p > {alpha_norm})."
                        else:
                             conclusion = f"❌ Rejeter H0. La distribution de `{selected_quant_col}` n'est probablement pas normale (p <= {alpha_norm})."
                        st.write(f"**Conclusion (seuil {alpha_norm*100}%)**: {conclusion}")
                    else:
                        st.warning(f"Le test de normalité '{test_type}' n'a pas pu être effectué (pas assez de données valides).")


                    # Graphiques de distribution
                    # --- Titres dynamiques ---
                    col1_dist, col2_dist = st.columns(2)
                    with col1_dist:
                        fig_hist, ax_hist = plt.subplots()
                        sns.histplot(clean_quant_data, kde=True, ax=ax_hist, color='skyblue')
                        # Ajouter la courbe normale théorique si moyenne/std sont valides
                        mean_val = clean_quant_data.mean()
                        std_val = clean_quant_data.std(ddof=1)
                        if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                            x_norm = np.linspace(clean_quant_data.min(), clean_quant_data.max(), 100)
                            ax_hist.plot(x_norm, norm.pdf(x_norm, mean_val, std_val), 'r--', label='Normale théorique')
                            ax_hist.legend()
                        ax_hist.set_title(f"Distribution de `{selected_quant_col}`")
                        st.pyplot(fig_hist)

                    with col2_dist:
                        fig_qq, ax_qq = plt.subplots()
                        # Vérifier qu'il y a assez de données pour probplot
                        if len(clean_quant_data) > 1:
                            probplot(clean_quant_data, plot=ax_qq)
                            ax_qq.set_title(f"Q-Q Plot pour `{selected_quant_col}`")
                        else:
                            ax_qq.text(0.5, 0.5, "Pas assez de données pour le Q-Q plot", ha='center', va='center')
                            ax_qq.set_title(f"Q-Q Plot pour `{selected_quant_col}`")
                        st.pyplot(fig_qq)

                    # Estimation statistique
                    # --- Sous-titre dynamique ---
                    st.subheader(f"Estimation par Intervalle de Confiance pour `{selected_quant_col}`")

                    # Décision méthode basée sur p-value valide
                    use_parametric = normality_test_done and p_value > alpha_norm

                    if use_parametric:
                        st.success(f"Méthode paramétrique (basée sur la loi de Student car `{selected_quant_col}` semble suivre une loi normale).")
                        n = len(clean_quant_data)
                        mean_val = clean_quant_data.mean()
                        std_err = stats.sem(clean_quant_data, nan_policy='omit') # Erreur standard de la moyenne

                        # IC Moyenne (Student)
                        if n > 1 and pd.notna(mean_val) and pd.notna(std_err) and std_err >= 0:
                             mean_ci = t.interval(0.95, df=n-1, loc=mean_val, scale=std_err)
                        else:
                             mean_ci = (np.nan, np.nan)

                        # IC Variance (Chi2)
                        var_val = clean_quant_data.var(ddof=1)
                        if n > 1 and pd.notna(var_val):
                            chi2_low_val = chi2.ppf(0.025, df=n-1)
                            chi2_high_val = chi2.ppf(0.975, df=n-1)
                            var_ci_low = (n-1)*var_val/chi2_high_val if chi2_high_val > 0 else 0
                            var_ci_upp = (n-1)*var_val/chi2_low_val if chi2_low_val > 0 else np.inf
                            var_ci = (var_ci_low, var_ci_upp)
                            # IC Ecart-type
                            sd_ci = (np.sqrt(var_ci[0]), np.sqrt(var_ci[1]))
                        else:
                            var_ci = (np.nan, np.nan)
                            sd_ci = (np.nan, np.nan)

                    else:
                        if not normality_test_done:
                            st.warning(f"Méthode non-paramétrique (Bootstrap) utilisée car le test de normalité n'a pas pu être effectué.")
                        else:
                            st.warning(f"Méthode non-paramétrique (Bootstrap) utilisée car `{selected_quant_col}` ne semble pas suivre une loi normale (p <= {alpha_norm}).")

                        n_resamples = st.slider("Nombre de réplications bootstrap", 100, 10000, 1000,
                                                key=f"n_resamples_{selected_quant_col}") # Clé unique

                        # Calcul des IC Bootstrap
                        mean_ci = bootstrap_ci(clean_quant_data, np.mean, n_resamples)
                        # Utiliser ddof=1 pour la variance/std d'échantillon dans bootstrap
                        var_ci = bootstrap_ci(clean_quant_data, np.var, n_resamples)
                        sd_ci = bootstrap_ci(clean_quant_data, np.std, n_resamples)

                        # Visualisation bootstrap (Exemple Écart-type)
                        # --- Titre dynamique ---
                        # Recalculer les échantillons bootstrap pour l'écart-type pour le graphique
                        if len(clean_quant_data) >= 2: # Nécessaire pour std
                            bootstrap_samples_sd = [np.std(np.random.choice(clean_quant_data.to_numpy(),
                                                                           size=len(clean_quant_data),
                                                                           replace=True), ddof=1)
                                                   for _ in range(n_resamples)]

                            fig_boot, ax_boot = plt.subplots()
                            sns.histplot(bootstrap_samples_sd, kde=True, ax=ax_boot, color='lightcoral')
                            # Ajouter lignes pour IC si valide
                            if pd.notna(sd_ci[0]) and pd.notna(sd_ci[1]):
                                ax_boot.axvline(sd_ci[0], color='red', linestyle='--', label=f'IC 95% Inf ({sd_ci[0]:.3f})')
                                ax_boot.axvline(sd_ci[1], color='red', linestyle='--', label=f'IC 95% Sup ({sd_ci[1]:.3f})')
                                ax_boot.legend()
                            ax_boot.set_title(f"Distribution Bootstrap de l'Écart-type (Éch.) pour `{selected_quant_col}`")
                            st.pyplot(fig_boot)
                        else:
                            st.info("Graphique Bootstrap non disponible (moins de 2 données valides).")


                    # Affichage des résultats d'estimation
                    st.markdown("---") # Séparateur visuel
                    st.write("**Estimations Ponctuelles et Intervalles de Confiance (95%)**")
                    col1_est, col2_est, col3_est = st.columns(3)

                    with col1_est:
                        st.metric(
                            "Moyenne",
                             # Afficher la valeur même si l'IC est nan
                            value=f"{clean_quant_data.mean():.4f}" if pd.notna(clean_quant_data.mean()) else "N/A",
                            delta=f"IC: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]" if pd.notna(mean_ci[0]) else "IC: N/A",
                            delta_color="off" # Pas de couleur pour l'IC
                        )

                    with col2_est:
                        st.metric(
                            "Variance (Éch.)",
                            value=f"{clean_quant_data.var(ddof=1):.4f}" if pd.notna(clean_quant_data.var(ddof=1)) else "N/A",
                            delta=f"IC: [{var_ci[0]:.4f}, {var_ci[1]:.4f}]" if pd.notna(var_ci[0]) else "IC: N/A",
                             delta_color="off"
                        )

                    with col3_est:
                        st.metric(
                            "Écart-type (Éch.)",
                             value=f"{clean_quant_data.std(ddof=1):.4f}" if pd.notna(clean_quant_data.std(ddof=1)) else "N/A",
                            delta=f"IC: [{sd_ci[0]:.4f}, {sd_ci[1]:.4f}]" if pd.notna(sd_ci[0]) else "IC: N/A",
                             delta_color="off"
                        )
                else:
                    # S'affiche si la colonne sélectionnée n'a que des NA
                    st.warning(f"Aucune donnée valide (non-NA) trouvée pour la variable `{selected_quant_col}`.")

        with tab2:
            # Partie Variables Qualitatives
            # Sélection de la variable qualitative
            qual_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            low_card_numeric = [col for col in df.select_dtypes(include=np.number).columns
                                if df[col].nunique() < 15 and df[col].nunique() > 1]
            # Ajouter les numériques faible cardinalité à la liste si pas déjà présents
            qual_cols += [col for col in low_card_numeric if col not in qual_cols]

            if not qual_cols:
                 # Ce header s'affiche seulement s'il n'y a pas de var. qualitatives
                st.header("Analyse des Variables Qualitatives")
                st.warning("Aucune variable qualitative (ou numérique à faible cardinalité) trouvée.")
            else:
                selected_qual_col = st.selectbox("Sélectionnez la variable qualitative", qual_cols,
                                                 key="qual_var_select")

                # --- Header dynamique ---
                st.header(f"Analyse de la Variable Qualitative : `{selected_qual_col}`")

                if selected_qual_col in df.columns:
                    # Traiter comme chaîne pour l'analyse des modalités, en gardant les NA pour le compte total initial
                    current_qual_data_with_na = df[selected_qual_col]
                    current_qual_data = current_qual_data_with_na.dropna().astype(str) # Convertir en str après dropna

                    # Sélection de la modalité
                    modalities = sorted(current_qual_data.unique())
                    if not modalities:
                         st.warning(f"Aucune modalité trouvée pour `{selected_qual_col}` après suppression des NA.")
                    else:
                        selected_modality = st.selectbox(f"Sélectionnez la modalité de `{selected_qual_col}` à analyser",
                                                        modalities, key=f"modality_select_{selected_qual_col}") # Clé unique

                        # --- Sous-titre dynamique ---
                        st.subheader(f"Analyse de la Proportion pour la modalité `{selected_modality}` (Variable `{selected_qual_col}`)")


                        # Paramètres d'analyse
                        conf_level_percent = st.slider("Niveau de confiance (%)", 90, 99, 95,
                                                key=f"conf_level_slider_{selected_qual_col}") # Clé unique
                        conf_level = conf_level_percent / 100.0

                        method_options = ['normal', 'agresti_coull', 'beta', 'wilson', 'jeffreys']
                        default_method_index = method_options.index('wilson')
                        method = st.selectbox("Méthode d'intervalle de confiance (proportion)",
                                                method_options, index=default_method_index,
                                                key=f"ci_method_prop_{selected_qual_col}", # Clé unique
                                                help="Méthodes de `statsmodels.stats.proportion.proportion_confint`. 'wilson' est souvent un bon choix.")


                        # Calcul des proportions basé sur les données NON-NA
                        total_count = len(current_qual_data) # Total des observations non-NA pour cette variable
                        modality_count = (current_qual_data == selected_modality).sum()

                        if total_count > 0:
                            proportion_val = modality_count / total_count
                            proportion_percent = proportion_val * 100

                            # Intervalle de confiance
                            try:
                                low, upp = proportion_confint(modality_count, total_count,
                                                            alpha=1-conf_level, method=method)
                                low_percent, upp_percent = low*100, upp*100
                            except Exception as e:
                                st.error(f"Erreur calcul IC: {e}")
                                low_percent, upp_percent = np.nan, np.nan

                            # Affichage des résultats
                            col1_qual, col2_qual = st.columns([1, 2]) # Donner plus de place au graphique

                            with col1_qual:
                                 # --- Texte de la métrique dynamique ---
                                st.metric(
                                    f"Proportion de '{selected_modality}'",
                                    value=f"{proportion_percent:.2f}%",
                                    delta=f"({modality_count} / {total_count} non-NA)", # Préciser le total
                                    delta_color="off"
                                )
                                if not np.isnan(low_percent):
                                    st.write(f"**Intervalle de confiance à {conf_level_percent:.0f}% ({method}):**")
                                    st.write(f"[{low_percent:.2f}%, {upp_percent:.2f}%]")
                                else:
                                     st.write(f"**Intervalle de confiance à {conf_level_percent:.0f}% ({method}):** Calcul impossible.")

                                # Téléchargement des résultats (nom de fichier dynamique)
                                results = pd.DataFrame({
                                    'Variable': [selected_qual_col],
                                    'Modalité Analysée': [selected_modality],
                                    'Proportion (%)': [proportion_percent],
                                    'Effectif Modalité': [modality_count],
                                    'Effectif Total (non-NA)': [total_count],
                                    'Méthode IC': [method],
                                    f'Borne Inférieure IC ({conf_level_percent:.0f}%)': [low_percent],
                                    f'Borne Supérieure IC ({conf_level_percent:.0f}%)': [upp_percent]
                                })

                                csv = results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label=f"Télécharger résultats pour '{selected_modality}'",
                                    data=csv,
                                    file_name=f'resultats_proportion_{selected_qual_col}_{selected_modality}.csv',
                                    mime='text/csv',
                                    key=f"download_prop_{selected_qual_col}_{selected_modality}" # Clé unique
                                )

                            with col2_qual:
                                # Graphique en camembert pour toutes les modalités de la variable sélectionnée
                                # --- Titre dynamique ---
                                st.write(f"**Répartition de toutes les modalités pour `{selected_qual_col}`**")
                                # Calculer les proportions sur les données non-NA
                                all_proportions = current_qual_data.value_counts(normalize=True) * 100
                                if not all_proportions.empty:
                                    fig_pie = px.pie(
                                        values=all_proportions.values,
                                        names=all_proportions.index,
                                        # title=f"Répartition de `{selected_qual_col}`", # Titre dans st.write au dessus
                                        color_discrete_sequence=px.colors.sequential.Viridis,
                                        hole=.3 # Petit trou au milieu pour un look "donut"
                                    )
                                    # Mettre en évidence la modalité sélectionnée
                                    pull_list = [0.2 if name == selected_modality else 0 for name in all_proportions.index]
                                    fig_pie.update_traces(textposition='inside', textinfo='percent+label', sort=False, pull=pull_list)
                                    fig_pie.update_layout(showlegend=False) # Légende redondante avec labels
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                else:
                                     st.warning(f"Pas de données à afficher dans le graphique pour `{selected_qual_col}`.")

                        else:
                            st.warning(f"Aucune donnée non nulle trouvée pour la variable `{selected_qual_col}` pour calculer les proportions.")

                        # Affichage des données brutes (optionnel, label dynamique)
                        if st.checkbox(f"Afficher les données brutes (non-NA) pour `{selected_qual_col}`", key=f"show_raw_data_{selected_qual_col}"): # Clé unique
                             st.dataframe(df[[selected_qual_col]].dropna())

                else:
                     # Ne devrait pas arriver si la sélection est basée sur df.columns
                    st.error(f"Erreur interne: La variable `{selected_qual_col}` n'a pas été trouvée.")

    # Message si le chargement a échoué après une tentative de téléversement
    elif uploaded_file is not None and df is None:
        st.error("Le chargement des données a échoué. Vérifiez le format ou le contenu du fichier.")

# Message si aucun fichier n'est encore téléversé
else:
    st.info("👋 Bienvenue ! Veuillez télécharger un fichier de données (CSV ou Excel) via la barre latérale pour commencer l'analyse.")
    st.markdown("Cette application vous permet d'analyser des variables quantitatives (calcul de statistiques descriptives, tests de normalité, intervalles de confiance paramétriques ou bootstrap) et qualitatives (calcul de proportions avec intervalles de confiance).")