import prometheus_client
print("Imported prometheus_client OK")
prometheus_client.start_http_server(9100)
print("Started server OK")
