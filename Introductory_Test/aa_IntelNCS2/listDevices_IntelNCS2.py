import openvino.runtime as ov
from openvino.runtime import get_version

core = ov.Core()

devices = core.available_devices

for device in devices:
    device_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
print(get_version())