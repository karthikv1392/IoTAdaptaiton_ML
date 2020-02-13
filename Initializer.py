_Author_ = "********"

# Initalizing all basic configurations

from configparser import ConfigParser
import traceback

CONFIG_FILE = "settings.conf"
CONFIG_SECTION = "settings"
CONFIG_SECTION_MODEL = "model" # For reading configurations specific to the model
CONFIG_SECTION_DT = "settings2" # For reading configurations specific to data traffic

class Initialize():
    def __init__(self):
        data_path = ""
        energy_val = 19160.0  # starting energy value as per CupCarbon
        component_count = 0  # The total number of sensors for which the monitoring needs to be done
        data_file = ""
        json_path = ""
        try:
            parser = ConfigParser()
            parser.read(CONFIG_FILE)
            self.data_path = parser.get(CONFIG_SECTION, "data_path")
            self.data_file = parser.get(CONFIG_SECTION, "data_file")
            self.json_path = parser.get(CONFIG_SECTION, "json_path")  # For storing the output time series object
            self.model_path = parser.get(CONFIG_SECTION, "model_path")  # For storing the output time series object
            self.energy_val = float(parser.get(CONFIG_SECTION, "initial_energy"))  # For getting maximum energy
            self.component_count = int(parser.get(CONFIG_SECTION, "component_count"))

            # For Model level
            self.epochs = int(parser.get(CONFIG_SECTION_MODEL, "epochs"))
            self.batch_size = int(parser.get(CONFIG_SECTION_MODEL, "batch_size"))
            self.num_features = int(parser.get(CONFIG_SECTION_MODEL, "num_features"))
            self.propotion_value = float(parser.get(CONFIG_SECTION_MODEL, "propotion_value"))


            # Data traffic related
            self.data_traffic_path = parser.get(CONFIG_SECTION_DT,"data_traffic_path")
            self.data_traffic_file = parser.get(CONFIG_SECTION_DT,"data_traffic_file")


        except Exception as e:
            traceback.print_exc()

