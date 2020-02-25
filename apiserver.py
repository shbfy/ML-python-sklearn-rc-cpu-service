# Import your handlers here
from service import Wine, Intro


# Configuration for web API implementation
def config(api):

    # Instantiate handlers and map routes
    api.add_route('/wine', Intro())
    api.add_route('/wine/{index:int(min=0)}', Wine())
