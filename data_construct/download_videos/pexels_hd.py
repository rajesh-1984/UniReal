import requests
import os
import time

def download_video(video_url, video_name):
    response = requests.get(video_url)
    if response.status_code == 200:
        with open(video_name, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {video_name}")
    else:
        print(f"Failed to download {video_name}")

def download_videos(api_key, query, save_root, max_results=1000):
    """
    Download HD landscape videos from Pexels API based on a search query.

    Parameters:
    - api_key: Your Pexels API key.
    - query: Search keyword (e.g., 'nature', 'city').
    - max_results: Maximum number of videos to download (default: 1000).
    """
    url = f'https://api.pexels.com/videos/search?query={query}&per_page=80&orientation=landscape'
    headers = {
        'Authorization': api_key
    }

    current_page = 1
    downloaded_videos = 0

    # Create directory to store downloaded videos
    folder_name =  os.path.join(save_root, query + '_hd_videos')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    data = {}
    while downloaded_videos < max_results:
        # Retry logic for API request
        retries = 5
        for attempt in range(retries):
            try:
                response = requests.get(f"{url}&page={current_page}", headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    break
                else:
                    raise Exception(f"Request failed with status code {response.status_code}")
            except Exception as e:
                print(f"Error fetching page {current_page}: {e}. Retrying {attempt+1}/{retries}...")
                time.sleep(2)
                if attempt == retries - 1:
                    print(f"Failed to fetch page {current_page} after {retries} attempts. Skipping...")
                    data = {}  # Skip this page
                    break

        if 'videos' not in data or len(data.get('videos', [])) == 0:
            print(f"No more videos found or unable to fetch page {current_page}.")
            break

        for video in data['videos']:
            try:
                if downloaded_videos >= max_results:
                    break

                video_files = video['video_files']

                # Filter HD quality videos only
                hd_videos = [vf for vf in video_files if vf.get('quality') == 'hd']

                if not hd_videos:
                    print(f"No HD version available for video {video['id']}")
                    continue

                # Use the first available HD version
                hd_video_url = hd_videos[0]['link']

                # Use the video ID from the URL as filename
                video_url = video['url']
                video_name = os.path.join(folder_name, f"{video_url[:-1].split('/')[-1]}.mp4")

                if os.path.exists(video_name):
                    print(f"Video already exists: {video_name}")
                    downloaded_videos += 1
                    continue

                download_video(hd_video_url, video_name)
                downloaded_videos += 1
            except Exception as e:
                print(f"Error downloading video {video['id']}: {e}")

        current_page += 1
        print(f"Downloaded {downloaded_videos} videos.")

    print("Download complete.")



coco_name_dict = {
    '1': 'person',
    '2': 'bicycle',
    '3': 'car',
    '4': 'motorbike',
    '5': 'aeroplane',
    '6': 'bus',
    '7': 'train',
    '8': 'truck',
    '9': 'boat',
    '10': 'trafficlight',
    '11': 'firehydrant',
    '12': 'streetsign',
    '13': 'stopsign',
    '14': 'parkingmeter',
    '15': 'bench',
    '16': 'bird',
    '17': 'cat',
    '18': 'dog',
    '19': 'horse',
    '20': 'sheep',
    '21': 'cow',
    '22': 'elephant',
    '23': 'bear',
    '24': 'zebra',
    '25': 'giraffe',
    '26': 'hat',
    '27': 'backpack',
    '28': 'umbrella',
    '29': 'shoe',
    '30': 'eyeglasses',
    '31': 'handbag',
    '32': 'tie',
    '33': 'suitcase',
    '34': 'frisbee',
    '35': 'skis',
    '36': 'snowboard',
    '37': 'sportsball',
    '38': 'kite',
    '39': 'baseballbat',
    '40': 'baseballglove',
    '41': 'skateboard',
    '42': 'surfboard',
    '43': 'tennisracket',
    '44': 'bottle',
    '45': 'plate',
    '46': 'wineglass',
    '47': 'cup',
    '48': 'fork',
    '49': 'knife',
    '50': 'spoon',
    '51': 'bowl',
    '52': 'banana',
    '53': 'apple',
    '54': 'sandwich',
    '55': 'orange',
    '56': 'broccoli',
    '57': 'carrot',
    '58': 'hotdog',
    '59': 'pizza',
    '60': 'donut',
    '61': 'cake',
    '62': 'chair',
    '63': 'sofa',
    '64': 'pottedplant',
    '65': 'bed',
    '66': 'mirror',
    '67': 'diningtable',
    '68': 'window',
    '69': 'desk',
    '70': 'toilet',
    '71': 'door',
    '72': 'tvmonitor',
    '73': 'laptop',
    '74': 'mouse',
    '75': 'remote',
    '76': 'keyboard',
    '77': 'cellphone',
    '78': 'microwave',
    '79': 'oven',
    '80': 'toaster',
    '81': 'book',
    '82': 'clock',
    '83': 'vase',
    '84': 'scissors',
    '85': 'teddybear',
    '86': 'hairdrier',
    '87': 'toothbrush',
    '88': 'hairbrush',
    '89': 'airplane',
    '90': 'ambulance',
    '91': 'backpack',
    '92': 'banana',
    '93': 'beachball',
    '94': 'bicycle',
    '95': 'blanket',
    '96': 'bookcase',
    '97': 'bottle',
    '98': 'box',
    '99': 'cabinet',
    '100': 'candle',
    '101': 'car',
    '102': 'carpet',
    '103': 'ceilingfan',
    '104': 'cereal',
    '105': 'chair',
    '106': 'clock',
    '107': 'coat',
    '108': 'computer',
    '109': 'couch',
    '110': 'cup',
    '111': 'curtain',
    '112': 'cushion',
    '113': 'desk',
    '114': 'deskchair',
    '115': 'dishwasher',
    '116': 'dog',
    '117': 'door',
    '118': 'drawer',
    '119': 'drum',
    '120': 'dustbin',
    '121': 'fan',
    '122': 'fence',
    '123': 'fishingpole',
    '124': 'flute',
    '125': 'football',
    '126': 'fridge',
    '127': 'fryingpan',
    '128': 'garden',
    '129': 'glove',
    '130': 'guitar',
    '131': 'hairdryer',
    '132': 'handbag',
    '133': 'hat',
    '134': 'headphones',
    '135': 'iron',
    '136': 'keyboard',
    '137': 'ladder',
    '138': 'lamp',
    '139': 'laptop',
    '140': 'lawnmower',
    '141': 'leash',
    '142': 'lightbulb',
    '143': 'magazine',
    '144': 'microwave',
    '145': 'mug',
    '146': 'musicstand',
    '147': 'newspaper',
    '148': 'notepad',
    '149': 'painting',
    '150': 'pan',
    '151': 'parasol',
    '152': 'pen',
    '153': 'piano',
    '154': 'picture',
    '155': 'plate',
    '156': 'plug',
    '157': 'remote',
    '158': 'refrigerator',
    '159': 'scissors',
    '160': 'sewingmachine',
    '161': 'shoppingcart',
    '162': 'skateboard',
    '163': 'socks',
    '164': 'suitcase',
    '165': 'sunhat',
    '166': 'swimsuit',
    '167': 'tablet',
    '168': 'telescope',
    '169': 'television',
    '170': 'thermometer',
    '171': 'toaster',
    '172': 'toiletpaper',
    '173': 'toothbrush',
    '174': 'towel',
    '175': 'trafficcone',
    '176': 'train',
    '177': 'tree',
    '178': 'tv',
    '179': 'umbrella',
    '180': 'vase',
    '181': 'wallet',
    '182': 'waterbottle',
    '183': 'whistle',
    '184': 'window',
    '185': 'wrench',
    '186': 'xylophone',
    '187': 'yoga mat',
    '188': 'zipper',
    '189': 'zoo',
    '190': 'zeppelin',
    '191': 'zucchini',
    '192': 'ball',
    '193': 'bedframe',
    '194': 'blanket',
    '195': 'boat',
    '196': 'bucket',
    '197': 'cactus',
    '198': 'camera',
    '199': 'car',
    '200': 'case'
}

def main():
    api_key = 'xxxx' # your API key 
    save_root = '/mnt/myworkspace/xic_space/projects/UniReal.open/data_construct/download_videos/save_videos' # your save root 
    for i in range(1,201):
        query = coco_name_dict[str(i)]
        download_videos(api_key, query, save_root)


if __name__ == "__main__":
    main()

