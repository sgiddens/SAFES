from synthesizer import DataSynthesizer
import compas.compas_preprocessing as compas_preprocessing

def define_settings():
    DP_settings_dict={
        # If True, use wrapper from snsynth. 
        # If False, use mbi FactoredInference directly.
        "use_snsynth_package": True, 
        # Required if using wrapper from snsynth
        "smart_noise_synthesizer": 'aim', 
        # Required if using mbi FactoredInference directly
        "cliques": 'all 2-way', 
    }

    fair_settings_dict={
        "y_label": "two_year_recid",
        "favorable_classes": [0],
        "protected_attribute_names": [
            'race-reduced-num', 
            'sex-num',
        ],
        "privileged_classes": [
            [1],
            [1],
        ],
        "categorical_features": [
            'age_cat-num',
            'c_charge_degree-num',
            'priors_count-reduced-num',
        ],
        "features_to_keep": [
            'age_cat-num',
            'c_charge_degree-num',
            'priors_count-reduced-num',
            'race-reduced-num',
            'sex-num',
            'two_year_recid',
        ],
        "metadata": {'label_maps': [{1.0: 1, 0.0: 0}],
                    'protected_attribute_maps': [
                        {1.0: 'Caucasian', 0.0: 'African-American'},
                        {1.0: 'Female', 0.0: 'Male'},
                    ]},
        "custom_distortion": 'compas',
        "verbose": False,
    }

    misc_settings_dict={
        "aif360_conversion": None,
        "protected_attribute_names_eval": [
            'race-reduced-num',
            'sex-num',
            'race-reduced-num*sex-num',
        ],
        "privileged_classes_eval": [
            [1],
            [1],
            [1],
        ]
    }

    return DP_settings_dict, fair_settings_dict, misc_settings_dict

def synthesizing_pipeline(epsilon_DP, epsilon_fair):
    DP_settings_dict, fair_settings_dict, misc_settings_dict = define_settings()

    # Get domain_dict
    df, domain_dict = compas_preprocessing.load_preprocessed_compas_data()
    DP_settings_dict["domain_dict"] = domain_dict
    
    # Synthesize data
    data_synthesizer = DataSynthesizer(epsilon_DP, epsilon_fair,
                                       DP_settings_dict, 
                                       fair_settings_dict,
                                       misc_settings_dict)
    synth_df = data_synthesizer.synthesize_DP_fair_df(df)
    return synth_df

