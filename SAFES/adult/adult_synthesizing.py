from synthesizer import DataSynthesizer
import adult.adult_preprocessing as adult_preprocessing

def define_settings():
    DP_settings_dict={
        # If True, use wrapper from snsynth. If False, use mbi FactoredInference directly
        "use_snsynth_package": True, 
        # Required if using wrapper from snsynth
        "smart_noise_synthesizer": 'aim', 
        # Required if using mbi FactoredInference directly
        "cliques": 'all 2-way', 
    }

    fair_settings_dict={
        "y_label": "income>50K",
        "favorable_classes": [1],
        "protected_attribute_names": [
            'race-reduced', 
            'sex-num',
        ],
        "privileged_classes": [
            [1],
            [1],
        ],
        "categorical_features": [
            'age-decade',
            'education-reduced'
        ],
        "features_to_keep": [
            'age-decade',
            'education-reduced',
            'race-reduced',
            'sex-num',
            'income>50K',
        ],
        "metadata": {'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                    'protected_attribute_maps': [
                        {1.0: 'White', 0.0: 'Non-white'},
                        {1.0: 'Male', 0.0: 'Female'},
                    ]},
        "custom_distortion": 'adult',
        "verbose": False,
    }

    misc_settings_dict={
        "aif360_conversion": 'adult', # Can also be a callable function or None
        "protected_attribute_names_eval": [
            'race-reduced',
            'sex-num',
            'race-reduced*sex-num',
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
    df, domain_dict = adult_preprocessing.load_preprocessed_adult_data()
    DP_settings_dict["domain_dict"] = domain_dict
    
    # Synthesize data
    data_synthesizer = DataSynthesizer(epsilon_DP, epsilon_fair,
                                       DP_settings_dict, 
                                       fair_settings_dict,
                                       misc_settings_dict)
    synth_df = data_synthesizer.synthesize_DP_fair_df(df)
    return synth_df

def draw_graph_pipeline(epsilon_DP):
    DP_settings_dict, fair_settings_dict, misc_settings_dict = define_settings()

    # Get domain_dict
    df, domain_dict = adult_preprocessing.load_preprocessed_adult_data()
    DP_settings_dict["domain_dict"] = domain_dict
    
    # Synthesize data
    data_synthesizer = DataSynthesizer(epsilon_DP, None,
                                       DP_settings_dict, 
                                       fair_settings_dict,
                                       misc_settings_dict)
    data_synthesizer.synthesize_DP_fair_df(df)
    data_synthesizer.draw_graphical_model("simulation_results/adult/graphs/")
    return