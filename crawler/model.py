# This code parses date/times, so please
#
#     pip install python-dateutil
#
# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = profile_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import Any, List, Optional, TypeVar, Callable, Type, cast
from datetime import datetime
from enum import Enum
from uuid import UUID
import dateutil.parser

T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_datetime(x: Any) -> datetime:
    return dateutil.parser.parse(x)


def from_bool(x: Any) -> bool:
    if isinstance(x, bool):
        x
    return False


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except Exception as ex:
            print(ex)
            pass
    # assert False


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


@dataclass
class SelectedInterest:
    id: str
    name: str

    @staticmethod
    def from_dict(obj: Any) -> 'SelectedInterest':
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        name = from_str(obj.get("name"))
        return SelectedInterest(id, name)

    def to_dict(self) -> dict:
        result: dict = {"id": from_str(self.id), "name": from_str(self.name)}
        return result


@dataclass
class UserInterests:
    selected_interests: List[SelectedInterest]

    @staticmethod
    def from_dict(obj: Any) -> 'UserInterests':
        assert isinstance(obj, dict)
        selected_interests = from_list(SelectedInterest.from_dict, obj.get("selected_interests"))
        return UserInterests(selected_interests)

    def to_dict(self) -> dict:
        result: dict = {"selected_interests": from_list(lambda x: to_class(SelectedInterest, x),
                                                        self.selected_interests)}
        return result


@dataclass
class ExperimentInfo:
    user_interests: UserInterests

    @staticmethod
    def from_dict(obj: Any) -> 'ExperimentInfo':
        assert isinstance(obj, dict)
        user_interests = UserInterests.from_dict(obj.get("user_interests"))
        return ExperimentInfo(user_interests)

    def to_dict(self) -> dict:
        result: dict = {"user_interests": to_class(UserInterests, self.user_interests)}
        return result


@dataclass
class Facebook:
    common_connections: List[Any]
    connection_count: int
    common_interests: List[Any]

    @staticmethod
    def from_dict(obj: Any) -> 'Facebook':
        assert isinstance(obj, dict)
        common_connections = from_list(lambda x: x, obj.get("common_connections"))
        connection_count = from_int(obj.get("connection_count"))
        common_interests = from_list(lambda x: x, obj.get("common_interests"))
        return Facebook(common_connections, connection_count, common_interests)

    def to_dict(self) -> dict:
        result: dict = {"common_connections": from_list(lambda x: x, self.common_connections),
                        "connection_count": from_int(self.connection_count),
                        "common_interests": from_list(lambda x: x, self.common_interests)}
        return result


@dataclass
class InstagramPhoto:
    image: str
    thumbnail: str
    ts: int

    @staticmethod
    def from_dict(obj: Any) -> 'InstagramPhoto':
        assert isinstance(obj, dict)
        image = from_str(obj.get("image"))
        thumbnail = from_str(obj.get("thumbnail"))
        ts = int(from_str(obj.get("ts")))
        return InstagramPhoto(image, thumbnail, ts)

    def to_dict(self) -> dict:
        result: dict = {"image": from_str(self.image), "thumbnail": from_str(self.thumbnail),
                        "ts": from_str(str(self.ts))}
        return result


@dataclass
class Instagram:
    last_fetch_time: datetime
    completed_initial_fetch: bool
    media_count: int
    photos: Optional[List[InstagramPhoto]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Instagram':
        assert isinstance(obj, dict)
        last_fetch_time = from_datetime(obj.get("last_fetch_time"))
        completed_initial_fetch = from_bool(obj.get("completed_initial_fetch"))
        media_count = from_int(obj.get("media_count"))
        photos = from_union([lambda x: from_list(InstagramPhoto.from_dict, x), from_none], obj.get("photos"))
        return Instagram(last_fetch_time, completed_initial_fetch, media_count, photos)

    def to_dict(self) -> dict:
        result: dict = {"last_fetch_time": self.last_fetch_time.isoformat(),
                        "completed_initial_fetch": from_bool(self.completed_initial_fetch),
                        "media_count": from_int(self.media_count), "photos": from_union(
                [lambda x: from_list(lambda x: to_class(InstagramPhoto, x), x), from_none], self.photos)}
        return result


@dataclass
class ProcessedFile:
    height: int
    width: int
    url: str

    @staticmethod
    def from_dict(obj: Any) -> 'ProcessedFile':
        assert isinstance(obj, dict)
        height = from_int(obj.get("height"))
        width = from_int(obj.get("width"))
        url = from_str(obj.get("url"))
        return ProcessedFile(height, width, url)

    def to_dict(self) -> dict:
        result: dict = {"height": from_int(self.height), "width": from_int(self.width),
                        "url": from_str(self.url)}
        return result


@dataclass
class Album:
    id: str
    name: str
    images: List[ProcessedFile]

    @staticmethod
    def from_dict(obj: Any) -> 'Album':
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        name = from_str(obj.get("name"))
        images = from_list(ProcessedFile.from_dict, obj.get("images"))
        return Album(id, name, images)

    def to_dict(self) -> dict:
        result: dict = {"id": from_str(self.id), "name": from_str(self.name),
                        "images": from_list(lambda x: to_class(ProcessedFile, x), self.images)}
        return result


@dataclass
class Track:
    id: str
    name: str
    album: Album
    artists: List[SelectedInterest]
    preview_url: str
    uri: str

    @staticmethod
    def from_dict(obj: Any) -> 'Track':
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        name = from_str(obj.get("name"))
        album = Album.from_dict(obj.get("album"))
        artists = from_list(SelectedInterest.from_dict, obj.get("artists"))
        preview_url = from_str(obj.get("preview_url"))
        uri = from_str(obj.get("uri"))
        return Track(id, name, album, artists, preview_url, uri)

    def to_dict(self) -> dict:
        result: dict = {"id": from_str(self.id), "name": from_str(self.name),
                        "album": to_class(Album, self.album),
                        "artists": from_list(lambda x: to_class(SelectedInterest, x), self.artists),
                        "preview_url": from_str(self.preview_url), "uri": from_str(self.uri)}
        return result


@dataclass
class SpotifyTopArtist:
    id: str
    name: str
    top_track: Track
    selected: bool

    @staticmethod
    def from_dict(obj: Any) -> 'SpotifyTopArtist':
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        name = from_str(obj.get("name"))
        top_track = Track.from_dict(obj.get("top_track"))
        selected = from_bool(obj.get("selected"))
        return SpotifyTopArtist(id, name, top_track, selected)

    def to_dict(self) -> dict:
        result: dict = {"id": from_str(self.id), "name": from_str(self.name),
                        "top_track": to_class(Track, self.top_track), "selected": from_bool(self.selected)}
        return result


@dataclass
class Teaser:
    string: str
    type: str

    @staticmethod
    def from_dict(obj: Any) -> 'Teaser':
        assert isinstance(obj, dict)
        string = from_str(obj.get("string"))
        type = obj.get("type") if "type" in obj else ""
        return Teaser(string, type)

    def to_dict(self) -> dict:
        result: dict = {"string": from_str(self.string),
                        "type": from_str(self.type)}
        return result


class ProfileType(Enum):
    USER = "user"


@dataclass
class Badge:
    type: str

    @staticmethod
    def from_dict(obj: Any) -> 'Badge':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        return Badge(type)

    def to_dict(self) -> dict:
        result: dict = {"type": from_str(self.type)}
        return result


@dataclass
class City:
    name: str

    @staticmethod
    def from_dict(obj: Any) -> 'City':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        return City(name)

    def to_dict(self) -> dict:
        result: dict = {"name": from_str(self.name)}
        return result


@dataclass
class Job:
    company: str
    title: str

    @staticmethod
    def from_dict(obj: Any) -> 'Job':
        assert isinstance(obj, dict)
        company = obj.get("company")['name'] if "company" in obj else ""
        title = obj.get("title")['name'] if "title" in obj else ""
        return Job(company, title)

    def to_dict(self) -> dict:
        result: dict = {"company": from_str(self.company), "title": from_str(self.title)}
        return result


@dataclass
class Algo:
    width_pct: float
    x_offset_pct: float
    height_pct: float
    y_offset_pct: float

    @staticmethod
    def from_dict(obj: Any) -> 'Algo':
        assert isinstance(obj, dict)
        width_pct = from_float(obj.get("width_pct"))
        x_offset_pct = from_float(obj.get("x_offset_pct"))
        height_pct = from_float(obj.get("height_pct"))
        y_offset_pct = from_float(obj.get("y_offset_pct"))
        return Algo(width_pct, x_offset_pct, height_pct, y_offset_pct)

    def to_dict(self) -> dict:
        result: dict = {"width_pct": to_float(self.width_pct), "x_offset_pct": to_float(self.x_offset_pct),
                        "height_pct": to_float(self.height_pct), "y_offset_pct": to_float(self.y_offset_pct)}
        return result


@dataclass
class CropInfo:
    processed_by_bullseye: bool
    user_customized: bool
    user: Optional[Algo] = None
    algo: Optional[Algo] = None

    @staticmethod
    def from_dict(obj: Any) -> 'CropInfo':
        assert isinstance(obj, dict)
        processed_by_bullseye = from_bool(obj.get("processed_by_bullseye"))
        user_customized = from_bool(obj.get("user_customized"))
        user = from_union([Algo.from_dict, from_none], obj.get("user"))
        algo = from_union([Algo.from_dict, from_none], obj.get("algo"))
        return CropInfo(processed_by_bullseye, user_customized, user, algo)

    def to_dict(self) -> dict:
        result: dict = {"processed_by_bullseye": from_bool(self.processed_by_bullseye),
                        "user_customized": from_bool(self.user_customized),
                        "user": from_union([lambda x: to_class(Algo, x), from_none], self.user),
                        "algo": from_union([lambda x: to_class(Algo, x), from_none], self.algo)}
        return result


@dataclass
class UserPhoto:
    id: UUID
    url: str
    processed_files: List[ProcessedFile]
    file_name: str
    processed_videos: Optional[List[ProcessedFile]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'UserPhoto':
        assert isinstance(obj, dict)
        id = UUID(obj.get("id"))
        url = from_str(obj.get("url"))
        processed_files = from_list(ProcessedFile.from_dict, obj.get("processedFiles"))
        file_name = from_str(obj.get("fileName"))
        processed_videos = from_union([lambda x: from_list(ProcessedFile.from_dict, x), from_none],
                                      obj.get("processedVideos"))
        return UserPhoto(id, url, processed_files, file_name, processed_videos)

    def to_dict(self) -> dict:
        result: dict = {"id": str(self.id),
                        "url": from_str(self.url),
                        "processedFiles": from_list(lambda x: to_class(ProcessedFile, x),
                                                    self.processed_files),
                        "fileName": from_str(self.file_name),
                        "processedVideos": from_union(
                            [lambda x: from_list(lambda x: to_class(ProcessedFile, x), x), from_none],
                            self.processed_videos)}
        return result


@dataclass
class User:
    id: str
    badges: List[Badge]
    bio: str
    birth_date: datetime
    name: str
    photos: List[UserPhoto]
    gender: int
    jobs: List[Job]
    schools: List[City]
    show_gender_on_profile: Optional[bool] = None
    city: Optional[City] = None
    is_traveling: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Any) -> 'User':
        assert isinstance(obj, dict)
        id = from_str(obj.get("_id"))
        badges = from_list(Badge.from_dict, obj.get("badges"))
        bio = from_str(obj.get("bio"))
        birth_date = from_datetime(obj.get("birth_date"))
        name = from_str(obj.get("name"))
        photos = from_list(UserPhoto.from_dict, obj.get("photos"))
        gender = from_int(obj.get("gender"))
        jobs = from_list(Job.from_dict, obj.get("jobs"))
        schools = from_list(City.from_dict, obj.get("schools"))
        show_gender_on_profile = from_union([from_bool, from_none], obj.get("show_gender_on_profile"))
        city = from_union([City.from_dict, from_none], obj.get("city"))
        is_traveling = from_union([from_bool, from_none], obj.get("is_traveling"))
        return User(id, badges, bio, birth_date, name, photos, gender, jobs, schools, show_gender_on_profile,
                    city, is_traveling)

    def to_dict(self) -> dict:
        result: dict = {"_id": from_str(self.id),
                        "badges": from_list(lambda x: to_class(Badge, x), self.badges),
                        "bio": from_str(self.bio), "birth_date": self.birth_date.isoformat(),
                        "name": from_str(self.name),
                        "photos": from_list(lambda x: to_class(UserPhoto, x), self.photos),
                        "gender": from_int(self.gender),
                        "jobs": from_list(lambda x: to_class(Job, x), self.jobs),
                        "schools": from_list(lambda x: to_class(City, x), self.schools),
                        "show_gender_on_profile": from_union([from_bool, from_none],
                                                             self.show_gender_on_profile),
                        "city": from_union([lambda x: to_class(City, x), from_none], self.city),
                        "is_traveling": from_union([from_bool, from_none], self.is_traveling)}
        return result


@dataclass
class ProfileElement:
    type: ProfileType
    user: User
    facebook: Facebook
    distance_mi: int
    content_hash: str
    s_number: int
    teaser: Teaser
    teasers: List[Teaser]
    is_superlike_upsell: bool
    experiment_info: Optional[ExperimentInfo] = None
    instagram: Optional[Instagram] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ProfileElement':
        assert isinstance(obj, dict)
        type = ProfileType(obj.get("type"))
        user = User.from_dict(obj.get("user"))
        facebook = Facebook.from_dict(obj.get("facebook"))
        distance_mi = from_int(obj.get("distance_mi"))
        content_hash = from_str(obj.get("content_hash"))
        s_number = from_int(obj.get("s_number"))
        teaser = Teaser.from_dict(obj.get("teaser"))
        teasers = from_list(Teaser.from_dict, obj.get("teasers"))
        is_superlike_upsell = from_bool(obj.get("is_superlike_upsell"))
        experiment_info = from_union([ExperimentInfo.from_dict, from_none], obj.get("experiment_info"))
        instagram = from_union([Instagram.from_dict, from_none], obj.get("instagram"))
        return ProfileElement(type, user, facebook, distance_mi, content_hash, s_number, teaser,
                              teasers, is_superlike_upsell, experiment_info, instagram)

    def to_dict(self) -> dict:
        result: dict = {"type": to_enum(ProfileType, self.type), "user": to_class(User, self.user),
                        "facebook": to_class(Facebook, self.facebook),
                        "distance_mi": from_int(self.distance_mi),
                        "content_hash": from_str(self.content_hash), "s_number": from_int(self.s_number),
                        "teaser": to_class(Teaser, self.teaser),
                        "teasers": from_list(lambda x: to_class(Teaser, x), self.teasers),
                        "is_superlike_upsell": from_bool(self.is_superlike_upsell),
                        "experiment_info": from_union([lambda x: to_class(ExperimentInfo, x), from_none],
                                                      self.experiment_info),
                        "instagram": from_union([lambda x: to_class(Instagram, x), from_none],
                                                self.instagram)}
        return result


def profile_from_dict(s: Any) -> List[ProfileElement]:
    return from_list(ProfileElement.from_dict, s)


def profile_to_dict(x: List[ProfileElement]) -> Any:
    return from_list(lambda x: to_class(ProfileElement, x), x)
