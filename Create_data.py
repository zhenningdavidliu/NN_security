import yaml
import sys
from data_bank import data_selector

if __name__ == '__main__':
    
    configfile = 'config_files/create_images_shades.yml'
    with open(configfile) as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    
    data_loader = data_selector(cgf['DATA']['name'], cgf['DATA']['arguments'])

    images, labels, diff = data_loader.load_data()
    print('Data done.')

    # Save the data
    
