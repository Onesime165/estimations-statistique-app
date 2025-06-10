import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, norm, t, chi2, probplot
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.proportion import proportion_confint

# --- Page Configuration ---
st.set_page_config(
    page_title="📊 Analyse Statistique",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark and Technological Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;500&display=swap');

    /* Main app styling */
    body {
        font-family: 'Roboto', sans-serif;
        color: #e0e6ed !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
        border-right: 2px solid #00ffff;
        box-shadow: 5px 0 15px rgba(0, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #b0c4de !important;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Main title */
    h1 {
        color: #00ffff;
        text-align: center;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }

    /* Sub-headers */
    h2, h3, h4 {
        color: #00ffff; /* Cyan accent for headers */
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #0f3460 0%, #16537e 100%);
        border-bottom: 2px solid #00ffff;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #b0c4de !important;
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: #00ffff !important;
        color: #0f0f23 !important;
        text-shadow: none;
    }

    /* Expander styling */
    .st-expander {
        border: 1px solid #00ffff;
        border-radius: 10px;
        background: rgba(15, 15, 35, 0.8);
    }
    .st-expander header {
        font-size: 1.2rem;
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    /* DataFrame styling */
    .stDataFrame {
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }

    /* Uploader styling in sidebar */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00ffff;
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
    }

    /* Author info box */
    .author-info {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# --- Plotting Theme Configuration (CORRIGÉ) ---
plt.style.use('dark_background')

# Configuration matplotlib corrigée
plt.rcParams.update({
    'figure.facecolor': '#0c0c0c',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#00ffff',
    'axes.labelcolor': '#e0e6ed',
    'xtick.color': '#e0e6ed',
    'ytick.color': '#e0e6ed',
    'grid.color': '#4a5568',
    'text.color': '#e0e6ed',
    'legend.facecolor': '#0f0f23',  # Couleur hexadécimale au lieu de rgba
    'legend.edgecolor': '#00ffff'
})

# Configuration du template Plotly
plotly_dark_template = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor='#1a1a2e', 
        paper_bgcolor='#0c0c0c', 
        font_color='#e0e6ed',
        xaxis=dict(gridcolor='#4a5568', linecolor='#e0e6ed'),
        yaxis=dict(gridcolor='#4a5568', linecolor='#e0e6ed'),
        title_font_color='#00ffff', 
        xaxis_title_font_color='#00ffff',
        yaxis_title_font_color='#00ffff',
        legend=dict(bgcolor='rgba(15,15,35,0.8)', bordercolor='#00ffff')
    )
)


# --- Application Title ---
st.title("📊 Application d'Analyse Statistique Interactive")
st.markdown("<p style='text-align: center; color: #b0c4de; font-family: Roboto, sans-serif;'>Une plateforme pour explorer vos données quantitatives et qualitatives avec des outils statistiques robustes.</p>", unsafe_allow_html=True)


# --- File uploader widget ---
uploaded_file = st.sidebar.file_uploader(
    "📤 Importer un fichier (CSV/Excel)",
    type=['csv', 'xlsx']
)

# Function to load data
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

# Bootstrap function
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
    
    if len(estimates) == 0: 
        return (np.nan, np.nan)
    
    alpha = 1 - conf_level
    return np.percentile(estimates, [100*alpha/2, 100*(1-alpha/2)])


# --- Main execution flow ---
df = load_data(uploaded_file)

if df is not None:
    st.sidebar.success(f"Fichier chargé: **{uploaded_file.name}**")
    st.sidebar.write(f"🔍 **{len(df)}** observations, **{len(df.columns)}** variables")

    tab1, tab2 = st.tabs(["**Variables Quantitatives**", "**Variables Qualitatives**"])

    with tab1:
        st.header("Analyse d'une Variable Quantitative")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            st.warning("Aucune variable quantitative trouvée dans les données.")
        else:
            selected_quant_col = st.selectbox("Sélectionnez la variable quantitative", numeric_cols, key="quant_var_select")
            st.subheader(f"Résultats pour : `{selected_quant_col}`")
            
            clean_quant_data = df[selected_quant_col].dropna().copy()

            if len(clean_quant_data) > 0:
                with st.expander("Statistiques Descriptives et Box Plot", expanded=True):
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
                        fig_box, ax_box = plt.subplots(figsize=(8, 6))
                        sns.boxplot(x=clean_quant_data, ax=ax_box, color='#00ffff')
                        ax_box.set_xlabel(selected_quant_col, color="#00ffff")
                        ax_box.set_title(f'Box Plot de `{selected_quant_col}`', color="#00ffff")
                        st.pyplot(fig_box)

                with st.expander("Vérification de la Normalité", expanded=True):
                    col1_dist, col2_dist = st.columns(2)
                    
                    with col1_dist:
                        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                        sns.histplot(clean_quant_data, kde=True, ax=ax_hist, color='#00ffff')
                        mean_val, std_val = clean_quant_data.mean(), clean_quant_data.std(ddof=1)
                        
                        if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                            x_norm = np.linspace(clean_quant_data.min(), clean_quant_data.max(), 100)
                            ax_hist.plot(x_norm, norm.pdf(x_norm, mean_val, std_val), 'r--', label='Normale théorique')
                            ax_hist.legend()
                        
                        ax_hist.set_title(f"Distribution de `{selected_quant_col}`")
                        st.pyplot(fig_hist)
                    
                    with col2_dist:
                        fig_qq, ax_qq = plt.subplots(figsize=(8, 6))
                        if len(clean_quant_data) > 1:
                            probplot(clean_quant_data, plot=ax_qq)
                            ax_qq.get_lines()[0].set_markerfacecolor('#00ffff')
                            ax_qq.get_lines()[0].set_markeredgecolor('#00ffff')
                            ax_qq.get_lines()[1].set_color('red')
                            ax_qq.set_title(f"Q-Q Plot pour `{selected_quant_col}`")
                        else:
                            ax_qq.text(0.5, 0.5, "Pas assez de données", ha='center')
                        st.pyplot(fig_qq)

                    test_type = st.radio("Type de test de normalité", 
                                       ["Shapiro-Wilk", "Kolmogorov-Smirnov"], 
                                       key=f"norm_test_type_{selected_quant_col}", 
                                       horizontal=True)
                    
                    p_value, stat, normality_test_done = np.nan, np.nan, False
                    
                    if test_type == "Shapiro-Wilk" and len(clean_quant_data) >= 3:
                        stat, p_value = shapiro(clean_quant_data)
                        normality_test_done = True
                    elif test_type == "Kolmogorov-Smirnov" and len(clean_quant_data) > 0:
                        stat, p_value = kstest(clean_quant_data, 'norm', args=(mean_val, std_val))
                        normality_test_done = True
                    
                    if normality_test_done:
                        st.write(f"**Test :** {test_type}, **Statistique :** {stat:.4f}, **P-value :** {p_value:.4f}")
                        if p_value > 0.05: 
                            st.success("✅ H0 non rejetée. La distribution semble normale.")
                        else: 
                            st.warning("❌ H0 rejetée. La distribution ne semble pas normale.")

                with st.expander("Estimation par Intervalle de Confiance", expanded=True):
                    use_parametric = normality_test_done and p_value > 0.05
                    
                    if use_parametric:
                        st.info("Méthode paramétrique (Student) utilisée.")
                        n = len(clean_quant_data)
                        mean_ci = t.interval(0.95, df=n-1, loc=clean_quant_data.mean(), scale=stats.sem(clean_quant_data))
                        
                        if n > 1 and pd.notna(clean_quant_data.var()):
                            chi2_low, chi2_high = chi2.ppf(0.025, df=n-1), chi2.ppf(0.975, df=n-1)
                            var_ci = ((n-1)*clean_quant_data.var()/chi2_high, (n-1)*clean_quant_data.var()/chi2_low)
                            sd_ci = (np.sqrt(var_ci[0]), np.sqrt(var_ci[1]))
                        else: 
                            var_ci, sd_ci = (np.nan, np.nan), (np.nan, np.nan)
                    else:
                        st.info("Méthode non-paramétrique (Bootstrap) utilisée.")
                        n_resamples = st.slider("Nombre de réplications bootstrap", 100, 5000, 1000, 
                                              key=f"n_resamples_{selected_quant_col}")
                        
                        mean_ci = bootstrap_ci(clean_quant_data, np.mean, n_resamples)
                        var_ci = bootstrap_ci(clean_quant_data, np.var, n_resamples)
                        sd_ci = bootstrap_ci(clean_quant_data, np.std, n_resamples)
                        
                        if len(clean_quant_data) >= 2:
                            bootstrap_samples_sd = [np.std(np.random.choice(clean_quant_data, size=len(clean_quant_data), replace=True), ddof=1) 
                                                  for _ in range(n_resamples)]
                            fig_boot, ax_boot = plt.subplots(figsize=(8, 6))
                            sns.histplot(bootstrap_samples_sd, kde=True, ax=ax_boot, color='#00ff88')
                            ax_boot.set_title(f"Distribution Bootstrap de l'Écart-type")
                            st.pyplot(fig_boot)
                    
                    st.write("---")
                    st.write("**Estimations et Intervalles de Confiance (95%)**")
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Moyenne", f"{clean_quant_data.mean():.4f}", 
                               f"IC: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]", delta_color="off")
                    col2.metric("Variance (Éch.)", f"{clean_quant_data.var(ddof=1):.4f}", 
                               f"IC: [{var_ci[0]:.4f}, {var_ci[1]:.4f}]", delta_color="off")
                    col3.metric("Écart-type (Éch.)", f"{clean_quant_data.std(ddof=1):.4f}", 
                               f"IC: [{sd_ci[0]:.4f}, {sd_ci[1]:.4f}]", delta_color="off")

    with tab2:
        st.header("Analyse d'une Variable Qualitative")
        qual_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        low_card_numeric = [col for col in df.select_dtypes(include=np.number).columns 
                           if 1 < df[col].nunique() < 15]
        qual_cols += [col for col in low_card_numeric if col not in qual_cols]

        if not qual_cols:
            st.warning("Aucune variable qualitative trouvée.")
        else:
            selected_qual_col = st.selectbox("Sélectionnez la variable qualitative", qual_cols, key="qual_var_select")
            st.subheader(f"Résultats pour : `{selected_qual_col}`")
            
            current_qual_data = df[selected_qual_col].dropna().astype(str)
            modalities = sorted(current_qual_data.unique())
            
            if not modalities:
                st.warning(f"Aucune modalité trouvée pour `{selected_qual_col}`.")
            else:
                selected_modality = st.selectbox(f"Sélectionnez la modalité à analyser", modalities, 
                                               key=f"modality_select_{selected_qual_col}")
                
                col1_qual, col2_qual = st.columns([1, 2])
                
                with col1_qual:
                    total_count = len(current_qual_data)
                    modality_count = (current_qual_data == selected_modality).sum()
                    
                    if total_count > 0:
                        proportion_percent = (modality_count / total_count) * 100
                        st.metric(f"Proportion de '{selected_modality}'", f"{proportion_percent:.2f}%", 
                                 f"{modality_count}/{total_count}", delta_color="off")
                        
                        conf_level = st.slider("Niveau de confiance (%)", 90, 99, 95, 
                                             key=f"conf_level_{selected_qual_col}") / 100.0
                        method = st.selectbox("Méthode d'IC", 
                                            ['wilson', 'normal', 'agresti_coull', 'beta', 'jeffreys'], 
                                            key=f"ci_method_{selected_qual_col}")
                        
                        low, upp = proportion_confint(modality_count, total_count, 
                                                    alpha=1-conf_level, method=method)
                        st.write(f"**IC à {conf_level*100:.0f}%:** [{low*100:.2f}%, {upp*100:.2f}%]")

                with col2_qual:
                    st.write(f"**Répartition de toutes les modalités**")
                    all_proportions = current_qual_data.value_counts(normalize=True) * 100
                    
                    if not all_proportions.empty:
                        # Créer une liste de couleurs dynamique basée sur la modalité sélectionnée
                        colors = ['#00ffff' if name == selected_modality else '#4a5568' for name in all_proportions.index]
                        
                        fig_pie = px.pie(values=all_proportions.values, names=all_proportions.index,
                                        color_discrete_sequence=colors, hole=.4)
                        
                        pull_list = [0.2 if name == selected_modality else 0 for name in all_proportions.index]
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label', 
                                            sort=False, pull=pull_list)
                        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), 
                                            template=plotly_dark_template)
                        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{selected_qual_col}_{selected_modality}")

else:
    st.info("👋 Bienvenue ! Veuillez télécharger un fichier de données (CSV ou Excel) pour commencer l'analyse.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="author-info">
        <h4>🧾 À propos de l'auteur</h4>
        <p><b>Nom:</b> N'dri</p>
        <p><b>Prénom:</b> Abo Onesime</p>
        <p><b>Rôle:</b> Data Analyst / Scientist</p>
        <p><b>Téléphone:</b> 07-68-05-98-87 / 01-01-75-11-81</p>
        <p><b>Email:</b> <a href="mailto:ndriablatie123@gmail.com" style="color:#00ff88;">ndriablatie123@gmail.com</a></p>
        <p><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/abo-onesime-n-dri-54a537200/" target="_blank" style="color:#00ff88;">Profil LinkedIn</a></p>
        <p><b>GitHub:</b> <a href="https://github.com/Aboonesime" target="_blank" style="color:#00ff88;">Mon GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
