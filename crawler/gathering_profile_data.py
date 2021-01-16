import json

from crawler.model import profile_from_dict, profile_to_dict


def gathering_profile():
    dict_profiles = {}
    list_profiles = []
    for index in range(0, 1000):
        with open(f"json/mydata{index}.json", 'rb') as fp:
            data = json.load(fp)
            results = data['data']['results']
            profiles = profile_from_dict(results)
            for profile in profiles:
                print(profile)
                _id = profile.user.id
                if _id not in dict_profiles:
                    dict_profiles[_id] = ""
                    list_profiles.append(profile)
    print(len(list_profiles))
    with open(f"../matcher/clustering/profiles.json", 'w', encoding='utf-8') as fp:
        json.dump(profile_to_dict(list_profiles), fp, ensure_ascii=False, indent=4)


gathering_profile()
