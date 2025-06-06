# test_app_simple.py
# this script doesn’t validate the prediction logic itself
# it simply ensures that both routes exist and respond successfully

import unittest
from app import app

class BasicAppTest(unittest.TestCase):
    def setUp(self):
        # Create a test client
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page_loads(self):
        """GET / should return status code 200."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_predict_endpoint(self):
        """
        POST /predict with a sample valid value should return 200.
        (Assumes model_1.pkl is present and returns something.)
        """
        response = self.client.post("/predict", data={"incentive": "10"})
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
