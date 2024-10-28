import yaml

with open("environment.yml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

# print(environment_data['dependencies'])
# print(environment_data["dependencies"][-1]['pip'])

with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"][:-1]:
        package_name, package_version, _ = dependency.split("=")
        file_handle.write("{}=={}\n".format(package_name, package_version))

    for dependency in environment_data["dependencies"][-1]['pip']:
        file_handle.write(f"{dependency}\n")