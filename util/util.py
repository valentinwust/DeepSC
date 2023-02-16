from datetime import datetime

def printwtime(text):
    """ Print text with current time.
    """
    print(datetime.now().strftime("%H:%M:%S"),"-",text)