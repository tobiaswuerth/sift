from hypothesis import settings

settings.register_profile("sift", derandomize=True, database=None, deadline=None, print_blob=True)
settings.load_profile("sift")
