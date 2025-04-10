
if __name__=='__main__':

    from configparser import ConfigParser
    
    from code.Pipeline import Pipeline

    config = ConfigParser()
    config.read('config.ini')
    pipeline = Pipeline(config=config)
    pipeline.run()
