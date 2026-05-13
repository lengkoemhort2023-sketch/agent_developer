from rest_framework import status
from rest_framework.response import Response


def success_response(message="Success", data=None, status_code=status.HTTP_200_OK):
    response = Response(
        {
            "result": "Success",
            "status_code": status_code,
            "result_message": message,
            "body": data,
        },
        status=status_code,
    )
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


def error_response(
    message,
    data=None,
    status_code=status.HTTP_400_BAD_REQUEST,
    error_code=None,
):
    response_data = {
        "result": "Error",
        "status_code": status_code,
        "result_message": message,
        "body": data,
    }
    if error_code is not None:
        response_data["error_code"] = error_code
    response = Response(response_data, status=status_code)
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response
