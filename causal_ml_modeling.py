"""
Causal Machine Learning Modeling
Implements sophisticated ML techniques for causal inference and prediction:
- Causal forests
- Double machine learning
- Meta-learners
- Instrumental variable approaches
- Counterfactual analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CausalMLModeler:
    def __init__(self, merged_file='merged.csv'):
        """Initialize with merged dataset for causal ML"""
        self.df = pd.read_csv(merged_file)
        self.df = self.df.dropna()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for causal ML analysis"""
        print(f"Loaded {len(self.df)} songs for causal ML analysis")
        
        # Define treatment and outcome variables
        self.treatment_var = 'high_energy'
        self.outcome_var = 'chart_success'
        
        # Create binary treatment (high energy vs low energy)
        energy_threshold = self.df['energy'].median()
        self.df[self.treatment_var] = (self.df['energy'] > energy_threshold).astype(int)
        
        # Create binary outcome (top 10 hit)
        self.df[self.outcome_var] = (self.df['chart_position'] <= 10).astype(int)
        
        # Define confounders
        self.confounders = [
            'danceability', 'valence', 'tempo', 'acousticness', 
            'instrumentalness', 'speechiness', 'loudness',
            'lexical_diversity', 'vader_compound', 'word_count'
        ]
        
        # Ensure all confounders exist
        self.confounders = [c for c in self.confounders if c in self.df.columns]
        
        print(f"Treatment: {self.treatment_var}")
        print(f"Outcome: {self.outcome_var}")
        print(f"Confounders: {len(self.confounders)} variables")
    
    def propensity_score_methods(self):
        """Implement various propensity score methods"""
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        
        X = self.df[self.confounders]
        treatment = self.df[self.treatment_var]
        outcome = self.df[self.outcome_var]
        
        # Different propensity score models
        ps_methods = {}
        
        # Logistic regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X, treatment)
        ps_methods['logistic'] = lr.predict_proba(X)[:, 1]
        
        # Gradient boosting
        gb = GradientBoostingClassifier(random_state=42, max_depth=3)
        gb.fit(X, treatment)
        ps_methods['boosting'] = gb.predict_proba(X)[:, 1]
        
        # Random forest
        rf = RandomForestClassifier(random_state=42, max_depth=5)
        rf.fit(X, treatment)
        ps_methods['random_forest'] = rf.predict_proba(X)[:, 1]
        
        # Evaluate propensity score models
        plt.figure(figsize=(15, 5))
        
        for i, (method, scores) in enumerate(ps_methods.items()):
            plt.subplot(1, 3, i+1)
            
            # Plot distributions by treatment group
            treated_scores = scores[treatment == 1]
            control_scores = scores[treatment == 0]
            
            plt.hist(treated_scores, alpha=0.5, label='Treated', bins=30, density=True)
            plt.hist(control_scores, alpha=0.5, label='Control', bins=30, density=True)
            plt.xlabel('Propensity Score')
            plt.ylabel('Density')
            plt.title(f'{method.title()} Propensity Scores')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('propensity_score_methods.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Use best method (based on visual overlap)
        best_method = 'boosting'  # Could be determined algorithmically
        propensity_scores = ps_methods[best_method]
        
        return propensity_scores, best_method
    
    def causal_forest_analysis(self):
        """Implement causal forest for heterogeneous treatment effects"""
        
        try:
            from econml.grf import CausalForest
        except ImportError:
            print("econml not available. Install with: pip install econml")
            return None
        
        X = self.df[self.confounders].values
        treatment = self.df[self.treatment_var].values
        outcome = self.df[self.outcome_var].values
        
        # Train causal forest
        cf = CausalForest(n_estimators=100, max_depth=5, min_samples_leaf=10, 
                         random_state=42)
        cf.fit(X, treatment, outcome)
        
        # Get heterogeneous treatment effects
        cate = cf.predict(X)
        
        # Add CATE to dataframe
        self.df['cate'] = cate
        
        # Analyze heterogeneous effects
        plt.figure(figsize=(15, 5))
        
        # CATE distribution
        plt.subplot(1, 3, 1)
        plt.hist(cate, bins=30, alpha=0.7, color='skyblue')
        plt.xlabel('Conditional Average Treatment Effect')
        plt.ylabel('Frequency')
        plt.title('Distribution of HTE')
        
        # CATE by decade
        plt.subplot(1, 3, 2)
        decade_cate = self.df.groupby('decade')['cate'].mean()
        plt.bar(decade_cate.index, decade_cate.values, alpha=0.7, color='coral')
        plt.xlabel('Decade')
        plt.ylabel('Mean CATE')
        plt.title('HTE by Decade')
        plt.xticks(rotation=45)
        
        # CATE vs key confounder
        plt.subplot(1, 3, 3)
        plt.scatter(self.df['danceability'], cate, alpha=0.5)
        plt.xlabel('Danceability')
        plt.ylabel('CATE')
        plt.title('HTE vs Danceability')
        
        plt.tight_layout()
        plt.savefig('causal_forest_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cf, cate
    
    def double_machine_learning(self):
        """Implement double machine learning for causal inference"""
        
        X = self.df[self.confounders]
        treatment = self.df[self.treatment_var]
        outcome = self.df[self.outcome_var]
        
        # First stage: predict treatment and outcome
        # Treatment model
        treatment_model = GradientBoostingRegressor(random_state=42, max_depth=3)
        treatment_model.fit(X, treatment)
        treatment_pred = treatment_model.predict(X)
        
        # Outcome model
        outcome_model = GradientBoostingRegressor(random_state=42, max_depth=3)
        outcome_model.fit(X, outcome)
        outcome_pred = outcome_model.predict(X)
        
        # Calculate residuals
        treatment_residual = treatment - treatment_pred
        outcome_residual = outcome - outcome_pred
        
        # Second stage: regress outcome residuals on treatment residuals
        # This gives the causal effect
        causal_effect = np.sum(treatment_residual * outcome_residual) / np.sum(treatment_residual ** 2)
        
        # Bootstrap confidence intervals
        bootstrap_effects = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(X), size=len(X), replace=True)
            
            X_boot = X.iloc[indices]
            treatment_boot = treatment.iloc[indices]
            outcome_boot = outcome.iloc[indices]
            
            # Refit models
            treatment_model_boot = GradientBoostingRegressor(random_state=42, max_depth=3)
            treatment_model_boot.fit(X_boot, treatment_boot)
            treatment_pred_boot = treatment_model_boot.predict(X_boot)
            
            outcome_model_boot = GradientBoostingRegressor(random_state=42, max_depth=3)
            outcome_model_boot.fit(X_boot, outcome_boot)
            outcome_pred_boot = outcome_model_boot.predict(X_boot)
            
            # Calculate causal effect
            treatment_residual_boot = treatment_boot - treatment_pred_boot
            outcome_residual_boot = outcome_boot - outcome_pred_boot
            
            if np.sum(treatment_residual_boot ** 2) > 0:
                effect_boot = np.sum(treatment_residual_boot * outcome_residual_boot) / np.sum(treatment_residual_boot ** 2)
                bootstrap_effects.append(effect_boot)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)
        
        # Visualize results
        plt.figure(figsize=(12, 8))
        
        # Residual plots
        plt.subplot(2, 2, 1)
        plt.scatter(treatment_pred, treatment_residual, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Treatment')
        plt.ylabel('Treatment Residuals')
        plt.title('Treatment Model Residuals')
        
        plt.subplot(2, 2, 2)
        plt.scatter(outcome_pred, outcome_residual, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Outcome')
        plt.ylabel('Outcome Residuals')
        plt.title('Outcome Model Residuals')
        
        plt.subplot(2, 2, 3)
        plt.scatter(treatment_residual, outcome_residual, alpha=0.5)
        plt.xlabel('Treatment Residuals')
        plt.ylabel('Outcome Residuals')
        plt.title('Second Stage Regression')
        
        # Causal effect with CI
        plt.subplot(2, 2, 4)
        plt.bar(['DML Causal Effect'], [causal_effect], 
               yerr=[[causal_effect - ci_lower], [ci_upper - causal_effect]], 
               capsize=5, alpha=0.7, color='green')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Causal Effect')
        plt.title(f'DML Estimate: {causal_effect:.3f}\\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        
        plt.tight_layout()
        plt.savefig('double_machine_learning.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return causal_effect, (ci_lower, ci_upper), bootstrap_effects
    
    def meta_learner_analysis(self):
        """Implement S-Learner, T-Learner, and X-Learner"""
        
        X = self.df[self.confounders]
        treatment = self.df[self.treatment_var]
        outcome = self.df[self.outcome_var]
        
        # S-Learner (Single learner)
        X_s = X.copy()
        X_s['treatment'] = treatment
        
        s_learner = GradientBoostingRegressor(random_state=42, max_depth=3)
        s_learner.fit(X_s, outcome)
        
        # Predict with and without treatment
        X_treated = X_s.copy()
        X_treated['treatment'] = 1
        X_control = X_s.copy()
        X_control['treatment'] = 0
        
        s_cate = s_learner.predict(X_treated) - s_learner.predict(X_control)
        
        # T-Learner (Two learners)
        # Treated group model
        X_treated_train = X[treatment == 1]
        y_treated_train = outcome[treatment == 1]
        
        treated_model = GradientBoostingRegressor(random_state=42, max_depth=3)
        treated_model.fit(X_treated_train, y_treated_train)
        
        # Control group model
        X_control_train = X[treatment == 0]
        y_control_train = outcome[treatment == 0]
        
        control_model = GradientBoostingRegressor(random_state=42, max_depth=3)
        control_model.fit(X_control_train, y_control_train)
        
        # T-Learner CATE
        t_cate = treated_model.predict(X) - control_model.predict(X)
        
        # X-Learner (More complex)
        # Step 1: Get imputed treatment effects
        treated_effects = y_treated_train - control_model.predict(X_treated_train)
        control_effects = treated_model.predict(X_control_train) - y_control_train
        
        # Step 2: Train effect models
        treated_effect_model = GradientBoostingRegressor(random_state=42, max_depth=3)
        treated_effect_model.fit(X_treated_train, treated_effects)
        
        control_effect_model = GradientBoostingRegressor(random_state=42, max_depth=3)
        control_effect_model.fit(X_control_train, control_effects)
        
        # Step 3: Get propensity scores and combine
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X, treatment)
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        # X-Learner CATE
        x_cate = (propensity_scores * treated_effect_model.predict(X) + 
                 (1 - propensity_scores) * control_effect_model.predict(X))
        
        # Compare meta-learners
        plt.figure(figsize=(15, 5))
        
        methods = ['S-Learner', 'T-Learner', 'X-Learner']
        cates = [s_cate, t_cate, x_cate]
        
        for i, (method, cate) in enumerate(zip(methods, cates)):
            plt.subplot(1, 3, i+1)
            plt.hist(cate, bins=30, alpha=0.7, label=method)
            plt.xlabel('CATE')
            plt.ylabel('Frequency')
            plt.title(f'{method} Treatment Effects')
            plt.axvline(x=np.mean(cate), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(cate):.3f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('meta_learner_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Add meta-learner results to dataframe
        self.df['s_cate'] = s_cate
        self.df['t_cate'] = t_cate
        self.df['x_cate'] = x_cate
        
        return {
            's_learner': (s_learner, s_cate),
            't_learner': (treated_model, control_model, t_cate),
            'x_learner': (treated_effect_model, control_effect_model, x_cate)
        }
    
    def instrumental_variable_analysis(self):
        """Implement instrumental variable approach"""
        
        # Create potential instrument: tempo-based energy proxy
        # High tempo songs are more likely to be high energy, but tempo might not directly affect chart success
        self.df['tempo_instrument'] = (self.df['tempo'] > self.df['tempo'].median()).astype(int)
        
        X = self.df[self.confounders]
        treatment = self.df[self.treatment_var]
        outcome = self.df[self.outcome_var]
        instrument = self.df['tempo_instrument']
        
        # First stage: regress treatment on instrument and confounders
        X_first_stage = X.copy()
        X_first_stage['instrument'] = instrument
        
        first_stage_model = LogisticRegression(random_state=42, max_iter=1000)
        first_stage_model.fit(X_first_stage, treatment)
        treatment_pred = first_stage_model.predict_proba(X_first_stage)[:, 1]
        
        # Second stage: regress outcome on predicted treatment
        X_second_stage = pd.DataFrame({
            'predicted_treatment': treatment_pred,
            **{col: X[col] for col in self.confounders}
        })
        
        second_stage_model = LogisticRegression(random_state=42, max_iter=1000)
        second_stage_model.fit(X_second_stage, outcome)
        
        # Extract IV estimate (coefficient on predicted treatment)
        iv_estimate = second_stage_model.coef_[0][0]  # First coefficient is for predicted treatment
        
        # Check instrument relevance (F-statistic)
        from sklearn.linear_model import LinearRegression
        relevance_model = LinearRegression()
        relevance_model.fit(X_first_stage, treatment)
        treatment_pred_linear = relevance_model.predict(X_first_stage)
        
        # Calculate F-statistic for instrument relevance
        ss_total = np.sum((treatment - np.mean(treatment)) ** 2)
        ss_residual = np.sum((treatment - treatment_pred_linear) ** 2)
        ss_explained = ss_total - ss_residual
        
        f_statistic = (ss_explained / 1) / (ss_residual / (len(treatment) - X_first_stage.shape[1] - 1))
        
        # Visualize IV analysis
        plt.figure(figsize=(12, 8))
        
        # First stage relationship
        plt.subplot(2, 2, 1)
        plt.scatter(instrument, treatment, alpha=0.5)
        plt.xlabel('Instrument (High Tempo)')
        plt.ylabel('Treatment (High Energy)')
        plt.title(f'First Stage: F-statistic = {f_statistic:.2f}')
        
        # Predicted vs actual treatment
        plt.subplot(2, 2, 2)
        plt.scatter(treatment, treatment_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        plt.xlabel('Actual Treatment')
        plt.ylabel('Predicted Treatment')
        plt.title('First Stage Fit')
        
        # IV estimate
        plt.subplot(2, 2, 3)
        plt.bar(['IV Estimate'], [iv_estimate], alpha=0.7, color='orange')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Causal Effect')
        plt.title(f'IV Estimate: {iv_estimate:.3f}')
        
        # Compare with OLS
        ols_model = LogisticRegression(random_state=42, max_iter=1000)
        ols_model.fit(X, treatment)
        ols_estimate = ols_model.coef_[0][0]
        
        plt.subplot(2, 2, 4)
        plt.bar(['OLS', 'IV'], [ols_estimate, iv_estimate], 
               alpha=0.7, color=['blue', 'orange'])
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Effect Estimate')
        plt.title('OLS vs IV Comparison')
        
        plt.tight_layout()
        plt.savefig('instrumental_variable_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return iv_estimate, f_statistic, (first_stage_model, second_stage_model)
    
    def counterfactual_analysis(self):
        """Generate counterfactual predictions"""
        
        X = self.df[self.confounders]
        treatment = self.df[self.treatment_var]
        outcome = self.df[self.outcome_var]
        
        # Train outcome model
        outcome_model = GradientBoostingRegressor(random_state=42, max_depth=3)
        X_with_treatment = X.copy()
        X_with_treatment['treatment'] = treatment
        outcome_model.fit(X_with_treatment, outcome)
        
        # Generate counterfactuals
        # What if all songs were high energy?
        X_all_treated = X.copy()
        X_all_treated['treatment'] = 1
        outcomes_all_treated = outcome_model.predict(X_all_treated)
        
        # What if all songs were low energy?
        X_all_control = X.copy()
        X_all_control['treatment'] = 0
        outcomes_all_control = outcome_model.predict(X_all_control)
        
        # Calculate average treatment effects
        ate_actual = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
        ate_counterfactual = outcomes_all_treated.mean() - outcomes_all_control.mean()
        
        # Visualize counterfactuals
        plt.figure(figsize=(15, 5))
        
        # Actual outcomes
        plt.subplot(1, 3, 1)
        plt.hist(outcome[treatment == 1], alpha=0.5, label='High Energy', bins=20, density=True)
        plt.hist(outcome[treatment == 0], alpha=0.5, label='Low Energy', bins=20, density=True)
        plt.xlabel('Chart Success')
        plt.ylabel('Density')
        plt.title(f'Actual Outcomes\\nATE: {ate_actual:.3f}')
        plt.legend()
        
        # Counterfactual outcomes
        plt.subplot(1, 3, 2)
        plt.hist(outcomes_all_treated, alpha=0.5, label='All High Energy', bins=20, density=True)
        plt.hist(outcomes_all_control, alpha=0.5, label='All Low Energy', bins=20, density=True)
        plt.xlabel('Predicted Chart Success')
        plt.ylabel('Density')
        plt.title(f'Counterfactual Outcomes\\nATE: {ate_counterfactual:.3f}')
        plt.legend()
        
        # Treatment effect distribution
        plt.subplot(1, 3, 3)
        individual_effects = outcomes_all_treated - outcomes_all_control
        plt.hist(individual_effects, bins=30, alpha=0.7, color='green')
        plt.xlabel('Individual Treatment Effect')
        plt.ylabel('Frequency')
        plt.title('Distribution of Individual Effects')
        
        plt.tight_layout()
        plt.savefig('counterfactual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'actual_ate': ate_actual,
            'counterfactual_ate': ate_counterfactual,
            'individual_effects': individual_effects,
            'outcomes_all_treated': outcomes_all_treated,
            'outcomes_all_control': outcomes_all_control
        }
    
    def run_causal_ml_analysis(self):
        """Run complete causal ML analysis pipeline"""
        print("Starting Causal Machine Learning Analysis...")
        print("=" * 50)
        
        print("\\n1. Propensity score methods...")
        propensity_scores, best_method = self.propensity_score_methods()
        
        print("\\n2. Causal forest analysis...")
        cf_results = self.causal_forest_analysis()
        
        print("\\n3. Double machine learning...")
        dml_results = self.double_machine_learning()
        
        print("\\n4. Meta-learner analysis...")
        meta_results = self.meta_learner_analysis()
        
        print("\\n5. Instrumental variable analysis...")
        iv_results = self.instrumental_variable_analysis()
        
        print("\\n6. Counterfactual analysis...")
        cf_analysis = self.counterfactual_analysis()
        
        print("\\nCausal ML analysis complete! Check generated PNG files.")
        
        return {
            'propensity_scores': (propensity_scores, best_method),
            'causal_forest': cf_results,
            'double_ml': dml_results,
            'meta_learners': meta_results,
            'instrumental_variable': iv_results,
            'counterfactuals': cf_analysis
        }

def main():
    """Main execution function"""
    
    # Check if merged data exists
    if not os.path.exists('merged.csv'):
        print("Error: merged.csv not found. Run merge_datasets.py first!")
        return
    
    # Run causal ML analysis
    modeler = CausalMLModeler()
    results = modeler.run_causal_ml_analysis()
    
    print("\\nCausal ML Analysis Summary:")
    print(f"- Propensity score method: {results['propensity_scores'][1]}")
    if results['double_ml']:
        print(f"- DML causal effect: {results['double_ml'][0]:.3f}")
    if results['instrumental_variable']:
        print(f"- IV causal effect: {results['instrumental_variable'][0]:.3f}")
    if results['counterfactuals']:
        print(f"- Counterfactual ATE: {results['counterfactuals']['counterfactual_ate']:.3f}")

if __name__ == "__main__":
    import os
    main()
