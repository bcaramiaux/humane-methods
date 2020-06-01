import json


def json_parser(config_file):

    if config_file == "":
        clients = [{'ip': '127.0.0.1', 'portout': 5009}]
        schedule = []
        for i in range(100):
            schedule.append({'episodes': [i], 'n_iterations': 100, 't_sleep': 0.0})
        env_config = {"step_size": 0.02}
        model = {"pre_trained_model": ""}
    
    else:
        config = json.load(open(config_file, 'r'))

        net = config['network']
        osc_server = net['server']
        osc_clients = net['clients']
        
        signals = config['signals']
        
        model_config = config['model']
        if "log" not in config.keys():
            logging = True
        else:
            logging = config["log"]

        schedule = config['schedule']
    
    return osc_server, osc_clients, signals, model_config, schedule, logging