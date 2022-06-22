
1. Environment Konfigurieren

    ENVIRONMENT.configs.config

2. PPO Alg Trainieren

    PPO_ALG.ppo.py

3. Umgebungsmodell lernen

    ENV_MODEL.train_env_model.py

4. I2A lernen

Entweder

    I2A_WITH_PPO.i2a_ppo.py (für ein einzelnen Run)

Oder

    I2A_WITH_PPO.run_experiment_grid.py (für Kombination mehrerer Parameter)

5. Ergebnisse Plotten

    PLOT.plot.py

6. Ergebnisse Testen

    TEST_POLICIES.test_policy.py

###################################################################

Im eingereichten Code sind trainierte Netze enthalten für:
PPO Algorithmus (PPO_ALG.data.ppo.2022-06-17-ppo-32000_time_pen)
ANN Model (ENV_MODEL.data.5_humans_EnvModel.2022-06-11_ANNEnvModel)
Mixed Model (ENV_MODEL.data.5_humans_EnvModel.2022-06-11_MixedEnvModel)
I2A Algorithmus (I2A_WITH_PPO.data.i2a)
