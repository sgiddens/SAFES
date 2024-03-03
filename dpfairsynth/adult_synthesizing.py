from synthesizer import DataSynthesizer
import adult_preprocessing

def define_settings():
    DP_settings_dict={
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

