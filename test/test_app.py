# Using py.test framework
from service import Intro, Wine


def test_example_message(client):
    """Example message should be returned"""
    client.app.add_route('/wine', Intro())

    result = client.simulate_get('/wine')
    assert result.json == {
        'message': 'This service verifies a model using the wine Test data set. '
                   'Invoke using the form /Wine/<index of test sample>. For example, /wine/24'}, \
        "The service test will fail until a trained model has been approved"


def test_classification_request(client):
    """Expected classification for Wine sample should be returned"""
    client.app.add_route('/wine/{index:int(min=0)}', Wine())

    result = client.simulate_get('/wine/1')
    assert result.status == "200 OK", "The service test will fail until a trained model has been approved"
    assert all(k in result.json for k in (
        "index", "predicted_label", "predicted")), "The service test will fail until a trained model has been approved"
