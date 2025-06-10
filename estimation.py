import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, norm, t, chi2, probplot
import plotly.express as px
from statsmodels.stats.proportion import proportion_confint

# --- Configuration de la page ---
st.set_page_config(
    layout="wide",
    page_title="📊 ESTIMATIONS STATISTIQUE",
    page_icon="📊"
)

# --- Styles CSS personnalisés ---
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
        color: #333;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: "Segoe UI", sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 20px;
    }
    .css-1offfwp.e1h7wlp61 {
        color: #2980b9;
    }
    .stButton button {
        background-color: #2980b9;
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
    .stMetric {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .author-info {
        background-color: #ffffff;
        border-left: 4px solid #2980b9;
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .author-info h4 {
        color: #2980b9;
        margin-bottom: 10px;
    }
    .author-info p {
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Widget déplacé en dehors de la fonction mise en cache ---
uploaded_file = st.sidebar.file_uploader("📤 Importer un fichier (CSV/Excel)", type=['csv', 'xlsx'])

# Fonction pour charger les données (prend maintenant uploaded_file en argument)
@st.cache_data
def load_data(uploaded_file_obj):
    if uploaded_file_obj is not None:
        try:
            file_name_ext = uploaded_file_obj.name
            if file_name_ext.endswith('.csv'):
                df_loaded = pd.read_csv(uploaded_file_obj)
            else:
                df_loaded = pd.read_excel(uploaded_file_obj)
            return df_loaded
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
            return None
    return None

# Fonction bootstrap (inchangée)
def bootstrap_ci(data, func, n_resamples=1000, conf_level=0.95):
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    if len(data) < 1 or (func in [np.var, np.std] and len(data) < 2):
        return (np.nan, np.nan)
    resamples = np.random.choice(data, size=(n_resamples, len(data)), replace=True)
    if func in [np.var, np.std]:
        estimates = np.array([func(resample, ddof=1) if len(resample) >= 2 else np.nan for resample in resamples])
        estimates = estimates[~np.isnan(estimates)]
        if len(estimates) < 2:
            return (np.nan, np.nan)
    else:
        estimates = np.apply_along_axis(func, 1, resamples)
    alpha = 1 - conf_level
    if len(estimates) == 0:
        return (np.nan, np.nan)
    return np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])

# --- Flux d'exécution principal ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.write(f"🔍 {len(df)} observations, {len(df.columns)} variables")
        tab1, tab2, tab3 = st.tabs(["Variables Quantitatives", "Variables Qualitatives", "À propos"])

        with tab1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.header(":variables: Analyse des Variables Quantitatives")
                st.warning("Aucune variable quantitative trouvée dans les données.")
            else:
                selected_quant_col = st.selectbox("Sélectionnez la variable quantitative", numeric_cols,
                                                  key="quant_var_select")
                st.markdown(f"<h2 style='color:#2980b9;'>:variables: `{selected_quant_col}`</h2>",
                            unsafe_allow_html=True)
                clean_quant_data = df[selected_quant_col].dropna().copy()
                if len(clean_quant_data) > 0:
                    col1_desc, col2_desc = st.columns(2)
                    with col1_desc:
                        stats_dict = {
                            "Moyenne": clean_quant_data.mean(),
                            "Médiane": clean_quant_data.median(),
                            "Écart-type (Éch.)": clean_quant_data.std(ddof=1),
                            "Variance (Éch.)": clean_quant_data.var(ddof=1),
                            "Minimum": clean_quant_data.min(),
                            "Maximum": clean_quant_data.max(),
                            "Effectif (non NA)": len(clean_quant_data)
                        }
                        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Valeur'])
                        st.dataframe(stats_df.style.format("{:.4f}"))
                    with col2_desc:
                        fig_box, ax_box = plt.subplots()
                        sns.boxplot(x=clean_quant_data, ax=ax_box, color='skyblue')
                        ax_box.set_xlabel(selected_quant_col)
                        ax_box.set_title(f'Box Plot de `{selected_quant_col}`')
                        st.pyplot(fig_box)

                    st.markdown(f"<h3>🧪 Test de normalité pour `{selected_quant_col}`</h3>", unsafe_allow_html=True)
                    test_type = st.radio("Type de test", ["Shapiro-Wilk", "Kolmogorov-Smirnov"],
                                         key=f"norm_test_type_{selected_quant_col}", horizontal=True)
                    p_value = np.nan
                    stat = np.nan
                    normality_test_done = False
                    if test_type == "Shapiro-Wilk" and len(clean_quant_data) >= 3:
                        stat, p_value = shapiro(clean_quant_data)
                        normality_test_done = True
                    elif test_type == "Kolmogorov-Smirnov":
                        stat, p_value = kstest(clean_quant_data, 'norm',
                                               args=(clean_quant_data.mean(), clean_quant_data.std(ddof=1)))
                        normality_test_done = True

                    if normality_test_done:
                        st.write(f"**Statistique:** {stat:.4f}")
                        st.write(f"**P-value:** {p_value:.4f}")
                        alpha_norm = 0.05
                        conclusion = f"✅ Ne pas rejeter H0. La distribution de `{selected_quant_col}` pourrait être normale." if p_value > alpha_norm else f"❌ Rejeter H0. La distribution de `{selected_quant_col}` ne semble pas normale."
                        st.write(f"**Conclusion (seuil {alpha_norm * 100}%):** {conclusion}")
                    else:
                        st.warning(f"Le test de normalité '{test_type}' n'a pas pu être effectué.")

                    col1_dist, col2_dist = st.columns(2)
                    with col1_dist:
                        fig_hist, ax_hist = plt.subplots()
                        sns.histplot(clean_quant_data, kde=True, ax=ax_hist, color='skyblue')
                        mean_val = clean_quant_data.mean()
                        std_val = clean_quant_data.std(ddof=1)
                        x_norm = np.linspace(clean_quant_data.min(), clean_quant_data.max(), 100)
                        ax_hist.plot(x_norm, norm.pdf(x_norm, mean_val, std_val), 'r--', label='Normale théorique')
                        ax_hist.legend()
                        ax_hist.set_title(f"Distribution de `{selected_quant_col}`")
                        st.pyplot(fig_hist)
                    with col2_dist:
                        fig_qq, ax_qq = plt.subplots()
                        if len(clean_quant_data) > 1:
                            probplot(clean_quant_data, plot=ax_qq)
                            ax_qq.set_title(f"Q-Q Plot pour `{selected_quant_col}`")
                        else:
                            ax_qq.text(0.5, 0.5, "Pas assez de données", ha='center', va='center')
                        st.pyplot(fig_qq)

                    st.markdown(f"<h3>🎯 Estimation par Intervalle de Confiance pour `{selected_quant_col}`</h3>",
                                unsafe_allow_html=True)
                    use_parametric = normality_test_done and p_value > 0.05
                    n_resamples = st.slider("Nombre de réplications bootstrap", 100, 10000, 1000,
                                            key=f"n_resamples_{selected_quant_col}")

                    if use_parametric:
                        n = len(clean_quant_data)
                        mean_val = clean_quant_data.mean()
                        var_val = clean_quant_data.var(ddof=1)
                        std_err = stats.sem(clean_quant_data)
                        mean_ci = t.interval(0.95, df=n - 1, loc=mean_val, scale=std_err) if n > 1 else (np.nan, np.nan)
                        chi2_low_val = chi2.ppf(0.025, df=n - 1)
                        chi2_high_val = chi2.ppf(0.975, df=n - 1)
                        var_ci = ((n - 1) * var_val / chi2_high_val, (n - 1) * var_val / chi2_low_val)
                        sd_ci = (np.sqrt(var_ci[0]), np.sqrt(var_ci[1]))
                    else:
                        mean_ci = bootstrap_ci(clean_quant_data, np.mean, n_resamples)
                        var_ci = bootstrap_ci(clean_quant_data, np.var, n_resamples)
                        sd_ci = bootstrap_ci(clean_quant_data, np.std, n_resamples)

                    col1_est, col2_est, col3_est = st.columns(3)
                    with col1_est:
                        st.metric("Moyenne",
                                  value=f"{clean_quant_data.mean():.4f}" if pd.notna(
                                      clean_quant_data.mean()) else "N/A",
                                  delta=f"IC: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]",
                                  delta_color="off")
                    with col2_est:
                        st.metric("Variance (Éch.)",
                                  value=f"{clean_quant_data.var(ddof=1):.4f}" if pd.notna(
                                      clean_quant_data.var(ddof=1)) else "N/A",
                                  delta=f"IC: [{var_ci[0]:.4f}, {var_ci[1]:.4f}]",
                                  delta_color="off")
                    with col3_est:
                        st.metric("Écart-type (Éch.)",
                                  value=f"{clean_quant_data.std(ddof=1):.4f}" if pd.notna(
                                      clean_quant_data.std(ddof=1)) else "N/A",
                                  delta=f"IC: [{sd_ci[0]:.4f}, {sd_ci[1]:.4f}]",
                                  delta_color="off")

                else:
                    st.warning(f"Aucune donnée valide (non-NA) trouvée pour la variable `{selected_quant_col}`.")

        with tab2:
            qual_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            low_card_numeric = [col for col in df.select_dtypes(include=np.number).columns if df[col].nunique() < 15]
            qual_cols += [col for col in low_card_numeric if col not in qual_cols]
            if not qual_cols:
                st.header(":chart_with_upwards_trend: Analyse des Variables Qualitatives")
                st.warning("Aucune variable qualitative trouvée.")
            else:
                selected_qual_col = st.selectbox("Sélectionnez la variable qualitative", qual_cols,
                                                key="qual_var_select")
                st.markdown(f"<h2 style='color:#2980b9;'>:chart_with_upwards_trend: `{selected_qual_col}`</h2>",
                            unsafe_allow_html=True)
                current_qual_data = df[selected_qual_col].dropna().astype(str)
                modalities = sorted(current_qual_data.unique())
                if not modalities:
                    st.warning(f"Aucune modalité trouvée pour `{selected_qual_col}`.")
                else:
                    selected_modality = st.selectbox(f"Sélectionnez la modalité de `{selected_qual_col}` à analyser",
                                                    modalities, key=f"modality_select_{selected_qual_col}")
                    st.markdown(f"<h3>🔎 Proportion pour la modalité `{selected_modality}`</h3>", unsafe_allow_html=True)
                    conf_level_percent = st.slider("Niveau de confiance (%)", 90, 99, 95,
                                                   key=f"conf_level_slider_{selected_qual_col}")
                    method_options = ['normal', 'agresti_coull', 'beta', 'wilson', 'jeffreys']
                    default_method_index = method_options.index('wilson')
                    method = st.selectbox("Méthode IC", method_options, index=default_method_index,
                                          key=f"ci_method_prop_{selected_qual_col}")
                    total_count = len(current_qual_data)
                    modality_count = (current_qual_data == selected_modality).sum()
                    if total_count > 0:
                        proportion_val = modality_count / total_count
                        proportion_percent = proportion_val * 100
                        try:
                            low, upp = proportion_confint(modality_count, total_count, alpha=1 - conf_level_percent / 100,
                                                          method=method)
                            low_percent, upp_percent = low * 100, upp * 100
                        except:
                            low_percent, upp_percent = np.nan, np.nan
                        col1_qual, col2_qual = st.columns([1, 2])
                        with col1_qual:
                            st.metric(f"'{selected_modality}'", value=f"{proportion_percent:.2f}%",
                                      delta=f"({modality_count}/{total_count})", delta_color="off")
                            st.write(f"**Intervalle de confiance ({conf_level_percent}%) ({method}) :**")
                            st.write(f"[{low_percent:.2f}%, {upp_percent:.2f}%]")
                        with col2_qual:
                            all_proportions = current_qual_data.value_counts(normalize=True) * 100
                            fig_pie = px.pie(values=all_proportions.values, names=all_proportions.index, hole=.3,
                                             color_discrete_sequence=px.colors.sequential.Viridis)
                            pull_list = [0.2 if name == selected_modality else 0 for name in all_proportions.index]
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label', sort=False,
                                                  pull=pull_list)
                            fig_pie.update_layout(showlegend=False)
                            st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("Aucune donnée non nulle trouvée.")

        with tab3:
            st.header("🧾 À Propos")
            st.markdown("""
<div class="author-info">
    <h4>🧾 About the Author</h4>
    <p><strong>Name:</strong> N'dri</p>
    <p><strong>First Name:</strong> Abo Onesime</p>
    <p><strong>Role:</strong> Data Analyst / Scientist</p>
    <p><strong>Phone:</strong> 07-68-05-98-87 / 01-01-75-11-81</p>
    <p><strong>Email:</strong> <a href="mailto:ndriablatie123@gmail.com">ndriablatie123@gmail.com</a></p>
    <p><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/abo-onesime-n-dri-54a537200/"  target="_blank">LinkedIn Profile</a></p>
    <p><strong>GitHub:</strong> <a href="https://github.com/Aboonesime"  target="_blank">My GitHub</a></p>
</div>
            """, unsafe_allow_html=True)

else:
    st.info("👋 Bienvenue ! Veuillez télécharger un fichier de données via la barre latérale.")
