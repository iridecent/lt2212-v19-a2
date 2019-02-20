import os


def get_subfoldernames(folder):
    topics = os.listdir(folder)
    return topics

print(get_subfoldernames("reuters-topics"))

