import json

def load_json(filename_json):
    """
    From the name of the json file,
    the dictionary is loaded.
    :param filename_json: string
    Json file name to load.
    :return: dict
    """
    with open(filename_json) as file_json:
        json_dict = json.load(file_json)

    return json_dict