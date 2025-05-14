import requests
from web_api import PredictionRequest, Modality


# Define the API endpoint
API_URL = "http://localhost:8000/predict"  # Update with your actual FastAPI endpoint

histo_image_url = "https://diagnostic-images-bucket.s3.me-central-1.amazonaws.com/ahmed_test/Kidney-49098.tif?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEgaDG1lLWNlbnRyYWwtMSJIMEYCIQDA1MqfnGCieJitjoO4Tc8PrtrooKT6iMkbrQf0OhlJRgIhAN%2BfON8vLeQR3dytjl3Qlm2bKpDc8YbDkqkV8zj6Cok1KtcDCLf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMNTM2Njk3MjU4ODcxIgzAoUjKySVe2PPUIkYqqwNey0Yd6UoXKGjLEj5D5DGG0ZuzAVej3dRMn%2BoZ1ixmiLK2L7mCh3wHqK4emFNojv1RxmvxoAEfFgbLv71e7sALQ3bMTkm0%2BBf84OGDOLv5KnbkvI02ibFTSBOBcd%2FujtAth1mjWhHpJOK7JFJ0tJ0Xaleu9ga3rIMRfYwRd75w5YtrmiIzfs7NWfJkp6DlgQzqGXI2L%2Bx6sSNx8JaDFMhzW9Fkh7hvNfnUG06Isy5bVUnJ6yf4vXiMn6v2mTNHrK1WCufGIHfR%2F54ZOQ6LDzlZ1LxVWfzRtP4ehPTu6aeiAi4%2FxT2D0fBRBtveFjcD%2F7trU76MfscuYvqvWVyGy9y8fhjK1O%2FRptW8kR6hUmePYdatNmaZvR8FQRgjC7d7TUhfAtfzelbHWIzeV23avSc9RJif8AlgBW3n1QU2vDD62GZdBJfRPqrCvB5NJOzS7H1Eu6Vlzaoz%2FPyeTTzFRigctPRt8msI8D7qZGvPuhN4WjFQm34zlnALeFrjyXWqp3U5DzTGwT7wpF3gMaSxOaFDVVCcQM%2FPipwd0l1B1F65VvIAH4MMb9vdRf0hMLu%2Fl8AGOt0CrskvnFgZTTLUzP2yFzeQqYpIk02AfmwnokoOYGLWoias7sGyGlBz98XAJCCtGfunUrfziRcyarKSeRXnb2DnKv%2Biw7pz2M5lcKx6LmrgseGZ5BJzrKk2ItFjrlNsGNpiHn%2BbECs%2BHHLxrk%2BrSfz3Q7BjK%2B7cM6MxT1NYDf6J1Of9sW2VaYVlpEhR85OZIHhtXYCt3ediQXRXRoCaDEJsb%2FuOAtOFSqMQ6hfkIQw8dfksZytNXWqsO0JU92DI%2FPhRKOXVfiP1Q8B3tvfri%2FYPVYb6mc8Lx69%2FA6OsHL4IyDsvRP5XXeZNF4dOVWf8Xf3S%2BK8VbXBVKsl5QqnsgaRRJuVLWfgGaBlINi2UgcEgYAcpA3EtpK3T15DhZY2nW0Xbm8kIAj2wNOO73d1JGfp%2FDk%2FYzI%2FkL7VvNhuqDbnIXSBbbQuOvWpHv8QHrYfNP9zzyx%2FTPfG5%2BdDabq0lrA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAXZ5NGIN3SO44HKKE%2F20250421%2Fme-central-1%2Fs3%2Faws4_request&X-Amz-Date=20250421T060514Z&X-Amz-Expires=36000&X-Amz-SignedHeaders=host&X-Amz-Signature=291d5361d54b368e0ee793ef91811a68ef529e521249b1b0ed7023b3e39779e6"

radio_image_url = "https://diagnostic-images-bucket.s3.me-central-1.amazonaws.com/ahmed_test/radiology.tiff?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjECYaDG1lLWNlbnRyYWwtMSJIMEYCIQDNni6e6H41Gx2%2BhMjNnIm%2BFbcYXABig66R9Lj9hx7zHAIhAN46fbrc61E00dgN1O%2FwTMReNIN56BmmOYMaDhHlBqoAKuADCHUQABoMNTM2Njk3MjU4ODcxIgzSAybP1aFHhb7bR%2BAqvQODIbpiKsO8YZYZRsMskg8MX%2FZJ2WgxWpHN1RBZE8pQQuo10DA3gk6rbhH4RBGP%2Flz5bzQvML1AQR64%2Fqc%2BlUWwRYrSnMBGhVl4NBOlabebn4b75XCgbD2WXl27%2F0k3U2gYoIoyyyxTSn2PnRsHLk%2F3jRzR8UyeKD6Zqw7%2FqdidvJDqZqorTYoVH5B8K7jAXRJtaOmhXZHM1WUj8QB2td4Z03q0Nhnh1dIS8EmKTCrPAKSZRoFdEqcuF2lkmTjFz9FPP9ZxrFFl%2BUuzbZtQhY8B983cDwsVOSihEqv57yzu7RWMYje7DZO5dMmLuMdvWE96wREWqibUSsOh82ZZaxDFUzm6r7EH9wWgIXXSHTKkL1JPqK91rpWF9xG4gnyyhdbHywcHlV8lmwECprlfeXeo5nWulf2nPU1Wji%2FHqCztVAqpLL1Yz1pkaJRfQsFRXvORk4QKduVHnRf79OVYPq4oGb8qDHqnKVAY%2BfsvtMCbRhwMVJ7X1Lc2p%2BKKkAxBJOiZMnj393fadITgR3LGo3euviIN9hd8Xrip42cQ2xwQRnHPVDlIogskIUpq8BotqcKWquIDcgiPhQtFkArfMOeun78GOuMC3lfZT2i1l1MZ5G5ko4hRFkaqNftSXuXabnI6125bLxFQq%2BXnsR2kJF7uu9p8wCSmMgRXUYbm%2BF4G6i73q0XLGE%2BJgpVyvTfKtJgl%2FUX3fzMxYT74adIHebJBcjqJK1LGQpTIiaIFeA9fogbFC4ip0y3zz5znZG5ubx3rAJRqCk530Q5T55a8VrtATJZ%2Fq0tcSPIPuHEvMwk5YqeJmUu5jiGeOV%2BJK5cL2MT63QC%2BEJkE76DmhnRLjgs%2FIOsaPoPkVea%2Bv8p4g3RkQWT3YGK9RSwuzwSTUMYBbAwvBwHcLEMv6eDhT9N1kNE7sWsZVY%2BPyNLIn%2F1VFypDL8a0Jh%2FLxoqy84u5gGkPKoVWeGhqfx5Xtgrqp1jWMVtvlrTsiynos36qxeJoGtC1CX4IIbHYY%2BAd02sVhUkZvQJ3qyLqAc%2BE2yWiJv6fuZWeMuVETL2j4vc8k5dkiseFN0dLz7jmHRWKJg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAXZ5NGIN37OH45RLA%2F20250329%2Fme-central-1%2Fs3%2Faws4_request&X-Amz-Date=20250329T112437Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=51d9b93091744586e5e0af48db42622c3ff41fc36b6846f17c1ae6ba1f5acc29"


def print_prediction_response(response_obj: dict):
    print("Patient Slide Id: ", response_obj["patient_slide_id"])
    print("Annotation Type: ", response_obj["annotation_type"])
    print("Annotations: ")

    def print_coordinate(message, coordinates):
        print(
            "       "
            + message
            + ": "
            + str(coordinates["x"])
            + ", "
            + str(coordinates["y"])
        )

    for idx, annotation in enumerate(response_obj["annotations"]):
        print(" " + str(idx) + ".")
        print("   name: ", annotation["name"])
        print("   biological_type: ", annotation["biological_type"])
        print("   confidence: ", annotation["confidence"])
        print("   shape: ", annotation["shape"])
        print("   description: ", annotation["description"])
        print("   coordinates: ")

        coordinates = annotation["coordinates"]
        print_coordinate("Top Left", coordinates[0])
        print_coordinate("Top Right", coordinates[1])
        print_coordinate("Bottom Right", coordinates[2])
        print_coordinate("Bottom Left", coordinates[3])


# Create request payload
payload = PredictionRequest(
    image_path=histo_image_url, patient_slide_id="1232123", modality=Modality.PATHOLOGY
).model_dump()

print("Sending Request for Pathology: ")
# Send POST request
response = requests.post(API_URL, json=payload)

# Print response
print("Status Code:", response.status_code)
print("Response:")
print_prediction_response(response.json())
print()

# Create request payload
#payload = PredictionRequest(
#    image_path=radio_image_url, patient_slide_id="1232123", modality=Modality.RADIOLOGY
#).model_dump()
#
#print("Sending Request for Radiology: ")
## Send POST request
#response = requests.post(API_URL, json=payload)
#
## Print response
#print("Status Code:", response.status_code)
#print("Response:")
#print_prediction_response(response.json())
