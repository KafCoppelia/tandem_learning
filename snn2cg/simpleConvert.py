import json

def genSimpleMapper(config, baseId):
    mapper = dict()
    coreIds = set()
    dests = set()
    for groupId in config['localPlace'].keys():
        localPlaceConfig = config['localPlace'][groupId]
        for coreId, coreConfig in localPlaceConfig.items():
            mapper[coreId] = coreId
    for coreId, coreConfig in config['relay'].items():
        mapper[coreId] = coreId
    return mapper

def genSimpleMapper16Core(config):
    mapper = dict()
    coreIds = set()
    dests = set()
    first_1_core = True


    places = {
        0,   1,  2,  3,
        32, 33, 34, 35,
        64, 65, 66, 67,
        96, 97, 98, 99
    }
    
    for groupId in config['localPlace'].keys():
        localPlaceConfig = config['localPlace'][groupId]
        if len(localPlaceConfig) == 8:
            print(localPlaceConfig.keys())
            for coreId, coreConfig in localPlaceConfig.items():
                rcoreId = int(coreId) % 1024
                mapper[coreId] = rcoreId
                places.remove(int(rcoreId))
        if len(localPlaceConfig) == 4:
            print(localPlaceConfig.keys())
            for coreId, coreConfig in localPlaceConfig.items():
                rcoreId = int(coreId) % 1024
                mapper[coreId] = int(rcoreId + 64)
                places.remove(int(rcoreId + 64))
        if len(localPlaceConfig) == 2:
            print(localPlaceConfig.keys())
            for coreId, coreConfig in localPlaceConfig.items():
                rcoreId = int(coreId) % 1024
                mapper[coreId] = int(rcoreId + 66)
                places.remove(int(rcoreId + 66))
        if len(localPlaceConfig) == 1:
            print(localPlaceConfig.keys())
            if first_1_core:
                first_1_core = False
                for coreId, coreConfig in localPlaceConfig.items():
                    rcoreId = int(coreId) % 1024
                    mapper[coreId] = int(rcoreId + 99)
                    places.remove(int(rcoreId + 99))
            else:
                for coreId, coreConfig in localPlaceConfig.items():
                    rcoreId = int(coreId) % 1024
                    for c in places:
                        mapper[coreId] = c
                        places.remove(c)
                        break
    assert len(places) == 0
    assert len(config['relay']) == 0
    return mapper

def gen16Core(config, baseId):
    cores = genSimpleMapper16Core(config)
    newCores = dict()
    for coreId, placeId in cores.items():
        newCores[coreId] = int(placeId + baseId)
    return newCores

if __name__ == "__main__":
    with open("config_EN2.json",'r') as f:
        config = json.load(f)
    mapper = genSimpleMapper(config)
    with open("mapper.txt",'w') as f:
        json.dump(mapper, f) 
    
