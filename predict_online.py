import requests
import os

label_dict = {0: 'CANCEL_PURCHASE',
              1: 'CUSTOMER_LOCATION',
              2: 'DELIVERY_TIMELINE',
              3: 'GENERAL_STATEMENT',
              4: 'IS_THE_VEHICLE_AVAILABLE',
              5: 'SCHEDULE_TEST_DRIVE',
              6: 'STORE_VEHICLE_LOCATION',
              7: 'WHAT_IS_PURCHASE_PENDING'}

if __name__ == '__main__':
    while True:
        text = input("Enter your text:\n")
        resp = requests.post('http://localhost:8501/v1/models/classifier:predict', json={"inputs": [text]})
        print('Loading prediction...')
        if resp.status_code == 200:
            predict_prob = resp.json()['outputs'][0]
            prediction_enc = predict_prob.index(max(predict_prob))
            prediction = label_dict[prediction_enc]
            print(prediction)
        else:
            print("Error, status: {}".format(resp.status_code))
