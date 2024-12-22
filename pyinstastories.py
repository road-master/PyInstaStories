# ruff: noqa: T201
from abc import abstractmethod
import argparse
import codecs
from dataclasses import dataclass
import datetime
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Generic, TypeVar
from xml.dom.minidom import parseString

import psutil

try:
    import urllib.request as urllib
except ImportError:
    import urllib as urllib

try:
    from instagram_private_api import (
        Client,
        ClientCookieExpiredError,
        ClientError,
        ClientLoginError,
        ClientLoginRequiredError,
    )
    from instagram_private_api import __version__ as client_version
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from instagram_private_api import (
        Client,
        ClientCookieExpiredError,
        ClientError,
        ClientLoginError,
        ClientLoginRequiredError,
    )
    from instagram_private_api import __version__ as client_version

script_version = "2.8"
python_version = sys.version.split(" ")[0]


def print_line() -> None:
    print("-" * 80)


# Commandline


class MyProgramArgs(argparse.Namespace):
    username: str
    password: str
    download: list[str]
    batchfile: Path | None
    takenat: bool
    novideothumbs: bool
    hqvideos: bool
    output: Path | None
    forcelogin: bool


class Commandline:
    def __init__(self) -> None:
        self.args = self.parse_args()
        self._usernames: list[str] | None = None
        self._hq_videos: bool | None = None

    def parse_args(self) -> type[MyProgramArgs]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-u",
            "--username",
            dest="username",
            type=str,
            required=False,
            help="Instagram username to login with.",
        )
        parser.add_argument(
            "-p",
            "--password",
            dest="password",
            type=str,
            required=False,
            help="Instagram password to login with.",
        )
        parser.add_argument(
            "-d",
            "--download",
            nargs="+",
            dest="download",
            type=str,
            required=False,
            help="Instagram user to download stories from.",
        )
        parser.add_argument(
            "-b,",
            "--batch-file",
            dest="batchfile",
            type=Path,
            required=False,
            help="Read a text file of usernames to download stories from.",
        )
        parser.add_argument(
            "-ta",
            "--taken-at",
            dest="takenat",
            action="store_true",
            help="Append the taken_at timestamp to the filename of downloaded items.",
        )
        parser.add_argument(
            "-nt",
            "--no-thumbs",
            dest="novideothumbs",
            action="store_true",
            help="Do not download video thumbnails.",
        )
        parser.add_argument(
            "-hqv",
            "--hq-videos",
            dest="hqvideos",
            action="store_true",
            help="Get higher quality video stories. Requires Ffmpeg.",
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output",
            type=Path,
            required=False,
            help="Destination folder for downloads.",
        )
        parser.add_argument("-f", "--force-login", dest="forcelogin", action="store_true", help="Force login.")

        # Workaround to 'disable' argument abbreviations
        parser.add_argument("--usernamx", help=argparse.SUPPRESS, metavar="IGNORE")
        parser.add_argument("--passworx", help=argparse.SUPPRESS, metavar="IGNORE")
        parser.add_argument("--downloax", help=argparse.SUPPRESS, metavar="IGNORE")
        parser.add_argument("--batch-filx", help=argparse.SUPPRESS, metavar="IGNORE")

        args, _ = parser.parse_known_args(namespace=MyProgramArgs)
        return args

    @property
    def usernames(self) -> list[str]:
        if not self._usernames:
            self._usernames = self.get_usernames()
        return self._usernames

    def get_usernames(self) -> list[str]:
        print(self.args.batchfile)
        if not self.args.download and not self.args.batchfile:
            print("[E] No usernames provided. Please use the -d or -b argument.")
            print_line()
            sys.exit(1)
        if self.args.download:
            return self.args.download
        if not self.args.batchfile or not self.args.batchfile.is_file():
            print("[E] The specified file does not exist.")
            print_line()
            sys.exit(1)
        usernames = [user.rstrip("\n") for user in self.args.batchfile.open()]
        if not usernames:
            print("[E] The specified file is empty.")
            print_line()
            sys.exit(1)
        print("[I] downloading {:d} users from batch file.".format(len(usernames)))
        print_line()
        return usernames

    @property
    def hq_videos(self) -> bool:
        if self._hq_videos is None:
            self._hq_videos = self.get_hq_videos()
        return self._hq_videos

    def get_hq_videos(self) -> bool:
        hq_videos = self.args.hqvideos
        if not hq_videos:
            return hq_videos
        if command_exists("ffmpeg"):
            print("[I] Downloading high quality videos enabled. Ffmpeg will be used.")
            print_line()
            return hq_videos
        print(
            "[W] Downloading high quality videos enabled but Ffmpeg could not be found. Falling back to default.",
        )
        print_line()
        return False


# Login


class ChallengeableClient:
    # Prevent HTTP 429 error
    REQUEST_INTERVAL = 1.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.client = Client(*args, **kwargs)

    def challenge(self, response: dict) -> None:
        print(response)
        challenge_url = response["challenge"]["url"]
        print(f"[E] Challenge required. Go to: {challenge_url}")
        input("After going above URL, press Enter to continue...")

    @property
    def settings(self) -> dict[str, Any]:
        # Reason: self.client.settings returns Any, but we know it's a dict.
        return self.client.settings  # type: ignore[no-any-return]

    @property
    def authenticated_user_id(self) -> str:
        # Reason: self.client.authenticated_user_id returns Any, but we know it's a str.
        return self.client.authenticated_user_id  # type: ignore[no-any-return]

    @property
    def cookie_jar(self) -> Any:
        return self.client.cookie_jar

    def user_info(self, user_id: str) -> dict:
        try:
            # Reason: self.client.user_info() returns Any, but we know it's a dict.
            return self.client.user_info(user_id)  # type: ignore[no-any-return]
        except ClientError as e:
            response = json.loads(e.error_response)
            if "message" not in response or response["message"] != "challenge_required":
                raise
            self.challenge(response)
        return self.user_info(user_id)

    def username_info(self, username: str) -> dict:
        try:
            # Reason: self.client.username_info() returns Any, but we know it's a dict.
            return self.client.username_info(username)  # type: ignore[no-any-return]
        except ClientError as e:
            response = json.loads(e.error_response)
            if "message" not in response or response["message"] != "challenge_required":
                raise
            self.challenge(response)
        return self.username_info(username)

    def user_story_feed(self, user_id: str) -> dict:
        try:
            # Reason: self.client.user_story_feed() returns Any, but we know it's a dict.
            return self.client.user_story_feed(user_id)  # type: ignore[no-any-return]
        except ClientError as e:
            response = json.loads(e.error_response)
            if "message" not in response or response["message"] != "challenge_required":
                raise
            self.challenge(response)
        return self.user_story_feed(user_id)

    def friendships_show(self, user_id: str) -> dict:
        try:
            # Reason: self.client.friendships_show() returns Any, but we know it's a dict.
            return self.client.friendships_show(user_id)  # type: ignore[no-any-return]
        except ClientError as e:
            response = json.loads(e.error_response)
            if "message" not in response or response["message"] != "challenge_required":
                raise
            self.challenge(response)
        return self.friendships_show(user_id)


class BytesSupportingJson:
    @classmethod
    def load(cls, path: Path, *, encoding: str = "utf-8") -> dict[str, Any]:
        # Reason: json.loads() returns Any, but we know it's a dict
        return json.loads(path.read_text(encoding=encoding), object_hook=cls.from_json)  # type: ignore[no-any-return]

    @staticmethod
    def from_json(json_object: dict[str, Any]) -> Any:
        if "__class__" in json_object and json_object.get("__class__") == "bytes":
            value: str = json_object["__value__"]
            return codecs.decode(value.encode(), "base64")
        return json_object

    @classmethod
    def dump(cls, path: Path, json_object: dict[str, Any]) -> None:
        path.write_text(json.dumps(json_object, default=cls.to_json))

    @staticmethod
    def to_json(python_object: Any) -> dict[str, Any]:
        if isinstance(python_object, bytes):
            return {"__class__": "bytes", "__value__": codecs.encode(python_object, "base64").decode()}
        raise TypeError(repr(python_object) + " is not JSON serializable")


class CredentialFile:
    def __init__(self, username: str = "") -> None:
        self.path_directory = Path.cwd() / ".pyinstastories_credentials"
        if not self.path_directory.is_dir():
            self.path_directory.mkdir()
        print(f"[I] Credencial will be saved to {self.path_directory}")
        print_line()
        self.path = self.get_settings_file(username)

    def get_settings_file(self, username: str) -> Path:
        if username == "":
            return next(self.path_directory.glob("credentials*.json"))
        return self.path_directory / Path(f"credentials_{username}.json")

    def save(self, api: ChallengeableClient) -> None:
        BytesSupportingJson.dump(self.path, api.settings)
        print(f"[I] New auth cookie file was made: {self.path}")

    def is_file(self) -> bool:
        return self.path.is_file()

    def load(self) -> dict[str, Any]:
        return BytesSupportingJson.load(self.path)

    def get_username(self) -> str:
        match = re.search("credentials_(.+?).json", self.path.name)
        if not match:
            msg = "Could not extract username from file name."
            raise ValueError(msg)
        return match.group(1)


class LoginExecutor:
    def __init__(self, settings_file: CredentialFile, username: str = "", password: str = "") -> None:
        self.settings_file = settings_file
        self.username = username
        self.password = password

    @abstractmethod
    def login(self) -> ChallengeableClient:
        raise NotImplementedError


class LoginNewExecutor(LoginExecutor):
    def login(self) -> ChallengeableClient:
        # settings file does not exist
        print(f"[W] Unable to find auth cookie file: {self.settings_file.path} (creating a new one...)")

        # login new
        return ChallengeableClient(self.username, self.password, on_login=lambda x: self.settings_file.save(x))


class LoginAgainExecutor(LoginExecutor):
    def __init__(self, settings_file: CredentialFile, username: str = "", password: str = "") -> None:
        super().__init__(settings_file, username, password)
        if not settings_file.is_file():
            self.device_id = None
        cached_settings = settings_file.load()
        self.device_id = cached_settings.get("device_id")

    def login(self) -> ChallengeableClient:
        api = ChallengeableClient(
            self.username,
            self.password,
            device_id=self.device_id,
            on_login=lambda x: self.settings_file.save(x),
        )
        print('[I] Re-login for "' + self.username + '".')
        return api


class LoginByCookieExecutor(LoginExecutor):
    def __init__(self, settings_file: CredentialFile, username: str = "", password: str = "") -> None:
        super().__init__(settings_file, username, password)
        self.cached_settings = settings_file.load()

    def login(self) -> ChallengeableClient:
        api = ChallengeableClient(self.username, self.password, settings=self.cached_settings)
        if self.username == "":
            try:
                self.username = self.settings_file.get_username()
            except AttributeError:
                self.username = api.authenticated_user_id
        print('[I] Using cached login cookie for "' + self.username + '".')
        return api


class LoginMethodSelector:
    def __init__(self, username: str = "", password: str = "", *, force_login: bool = False) -> None:
        self.username = username
        self.password = password
        self.force_login = force_login
        self.settings_file = CredentialFile(username)

    def login(self) -> ChallengeableClient:
        try:
            login_executor = self.choose_login_method()
            api = self.login_main(login_executor)
        except Exception as e:
            if str(e).startswith("unsupported pickle protocol"):
                print("[W] This cookie file is not compatible with Python {}.".format(sys.version.split(" ")[0][0]))
                print("[W] Please delete your cookie file 'credentials_{}.json' and try again.".format(self.username))
            else:
                print(f"[E] Unexpected Exception: {e}")
            print_line()
            sys.exit(99)
        print('[I] Login to "' + self.username + '" OK!')
        cookie_expiry = api.cookie_jar.auth_expires
        expirty_date = datetime.datetime.fromtimestamp(cookie_expiry).strftime("%Y-%m-%d at %I:%M:%S %p")
        print(f"[I] Login cookie expiry date: {expirty_date}")
        return api

    def choose_login_method(self) -> LoginExecutor:
        if not self.settings_file.is_file():
            print(f"[I] No auth cookie file found: {self.settings_file.path} LoginNewExecutor will be used.")
            return LoginNewExecutor(self.settings_file, self.username, self.password)
        if self.force_login:
            print("[I] Force login enabled, LoginAgainExecutor will be used.")
            return LoginAgainExecutor(self.settings_file, self.username, self.password)
        print(f"[I] Using cached login cookie: {self.settings_file.path} LoginByCookieExecutor will be used.")
        return LoginByCookieExecutor(self.settings_file, self.username, self.password)

    def try_login_first(self, login_executor: LoginExecutor) -> ChallengeableClient:
        try:
            return login_executor.login()
        except (ClientCookieExpiredError, ClientLoginRequiredError) as e:
            print(f"[E] ClientCookieExpiredError/ClientLoginRequiredError: {e}")
            # Login expired
            # Do relogin but use default ua, keys and such
            if not (self.username and self.password):
                print("[E] The login cookie has expired, but no login arguments were given.")
                print("[E] Please supply --username and --password arguments.")
                print_line()
                sys.exit(0)
        login_again_executor = LoginAgainExecutor(self.settings_file, self.username, self.password)
        return login_again_executor.login()

    def login_main(self, login_executor: LoginExecutor) -> ChallengeableClient:
        try:
            return self.try_login_first(login_executor)
        except ClientLoginError as e:
            print(
                "[E] Could not login: {:s}.\n[E] {:s}\n\n{:s}".format(
                    json.loads(e.error_response).get("error_title", "Error title not available."),
                    json.loads(e.error_response).get("message", "Not available"),
                    e.error_response,
                ),
            )
            print_line()
            sys.exit(9)
        except ClientError as e:
            print("[E] Client Error: {:s}".format(e.error_response))
            print_line()
            sys.exit(9)


def login_instagram(commandline: Commandline) -> ChallengeableClient:
    args = commandline.args

    force_login = args.forcelogin or False
    if force_login:
        print("[I] forceLogin = True")
    if args.username and args.password:
        return LoginMethodSelector(args.username, args.password, force_login=force_login).login()
    if args.username:
        return LoginMethodSelector(args.username, force_login=force_login).login()
    try:
        if not CredentialFile().is_file():
            print("[E] No username/password provided, but there is no login cookie present either.")
            print("[E] Please supply --username and --password arguments.")
            exit(1)
        return LoginMethodSelector().login()
    except:
        print("[E] Credentials file not found!")
        exit(1)


# Downloader


@dataclass
class User:
    id: str
    name: str


class UserFactory:
    def __init__(self, ig_client: ChallengeableClient) -> None:
        self.ig_client = ig_client

    def create(self, username: str) -> User:
        if username.isdigit():
            return self.create_by_user_id(username)
        return self.create_by_username(username)

    def create_by_user_id(self, user_id: str) -> User:
        user_info = self.ig_client.user_info(user_id)
        user = user_info.get("user")
        if not user:
            raise Exception("No user is associated with the given user id.")
        return User(user_id, user.get("username"))

    def create_by_username(self, username: str) -> User:
        user_res = self.ig_client.username_info(username)
        return User(user_res["user"]["pk"], username)


def command_exists(command: str) -> bool:
    try:
        fnull = open(os.devnull, "w")
        subprocess.call([command], stdout=fnull, stderr=subprocess.STDOUT)
        return True
    except OSError:
        return False


def download_file(url: str, path: Path, attempt: int = 0) -> None:
    try:
        urllib.urlretrieve(url, path)
        urllib.urlcleanup()
    except Exception as e:
        if attempt != 3:
            attempt += 1
            print("[E] ({:d}) Download failed: {:s}.".format(attempt, str(e)))
            print("[W] Trying again in 5 seconds.")
            time.sleep(5)
            download_file(url, path, attempt)
        else:
            print("[E] Retry failed three times, skipping file.")
            print_line()


@dataclass
class Media:
    url: str
    taken_ts: str | None
    taken_at: bool

    def get_file_name_mp4(self) -> str:
        file_name = self.url.split("/")[-1]
        if not self.taken_at:
            return file_name.split(".")[0] + ".mp4"
        if not self.taken_ts:
            file_name = file_name.split(".")[0] + ".mp4"
            print("[E] Could not determine timestamp filename for this file, using default: " + file_name)
            return file_name
        return self.taken_ts + ".mp4"

    def get_file_name_jpeg(self) -> str:
        file_name = (self.url.split("/")[-1]).split("?", 1)[0]
        if not self.taken_at:
            return file_name.split(".")[0] + ".jpg"
        if not self.taken_ts:
            return file_name.split(".")[0] + ".jpg"
        return self.taken_ts + ".jpg"


class AbstractFeedReelItemsCollector:
    def __init__(self, *, taken_at: bool, no_video_thumbs: bool = False) -> None:
        self.taken_at = taken_at
        self.no_video_thumbs = no_video_thumbs
        self.list_image: list[Media] = []

    def collect(self, media: dict) -> None:
        taken_ts = self.get_taken_ts(media)
        if "video_versions" in media:
            self.collect_video(media, taken_ts)
        if "image_versions2" in media:
            is_video = "video_versions" in media
            if (is_video and not self.no_video_thumbs) or not is_video:
                url = media["image_versions2"]["candidates"][0]["url"]
                self.list_image.append(Media(url, taken_ts, self.taken_at))

    def get_taken_ts(self, media: dict) -> str | None:
        if not self.taken_at:
            return None
        if media.get("imported_taken_at"):
            imported_taken_at = media.get("imported_taken_at", "")
            if imported_taken_at > 10000000000:
                imported_taken_at /= 1000
            return (
                datetime.datetime.utcfromtimestamp(media.get("taken_at", "")).strftime("%Y-%m-%d_%H-%M-%S")
                + "__"
                + datetime.datetime.utcfromtimestamp(imported_taken_at).strftime("%Y-%m-%d_%H-%M-%S")
            )
        return datetime.datetime.utcfromtimestamp(media.get("taken_at", "")).strftime("%Y-%m-%d_%H-%M-%S")

    @abstractmethod
    def collect_video(self, media: dict, taken_ts: str | None) -> None:
        raise NotImplementedError


class FFmpegFeedReelItemsCollector(AbstractFeedReelItemsCollector):
    def __init__(self, *, taken_at: bool, no_video_thumbs: bool = False) -> None:
        super().__init__(taken_at=taken_at, no_video_thumbs=no_video_thumbs)
        self.list_video_v: list[Media] = []
        self.list_video_a: list[str] = []

    def collect_video(self, media: dict, taken_ts: str | None) -> None:
        video_manifest = parseString(media["video_dash_manifest"])
        # video_period = video_manifest.documentElement.getElementsByTagName('Period')
        # video_representations = video_period[0].getElementsByTagName('Representation')
        # video_url = video_representations.pop().getElementsByTagName('BaseURL')[0].childNodes[0].nodeValue
        # audio_url = video_representations[0].getElementsByTagName('BaseURL')[0].childNodes[0].nodeValue
        video_period = video_manifest.documentElement.getElementsByTagName("Period")
        representations = video_period[0].getElementsByTagName("Representation")
        video_url = representations[0].getElementsByTagName("BaseURL")[0].childNodes[0].nodeValue
        audio_element = representations.pop()
        if audio_element.getAttribute("mimeType") == "audio/mp4":
            audio_url = audio_element.getElementsByTagName("BaseURL")[0].childNodes[0].nodeValue
        else:
            audio_url = "noaudio"
        self.list_video_v.append(Media(video_url, taken_ts, self.taken_at))
        self.list_video_a.append(audio_url)


class FeedReelItemsCollector(AbstractFeedReelItemsCollector):
    def __init__(self, *, taken_at: bool, no_video_thumbs: bool = False) -> None:
        super().__init__(taken_at=taken_at, no_video_thumbs=no_video_thumbs)
        self.list_video: list[Media] = []

    def collect_video(self, media: dict, taken_ts: str | None) -> None:
        self.list_video.append(Media(media["video_versions"][0]["url"], taken_ts, self.taken_at))


@dataclass
class SavePath:
    final: Path
    video: Path
    audio: Path


class FFmpegFinalizer:
    def __init__(self, save_path: SavePath) -> None:
        self.save_path = save_path

    def finalize(self) -> None:
        ffmpeg_binary = os.getenv("FFMPEG_BINARY", "ffmpeg")
        command = [ffmpeg_binary]
        command += self.get_arguments()
        exit_code = subprocess.call(command, stdout=None, stderr=subprocess.STDOUT)
        self.remove_temporary_files()
        if exit_code == 0:
            return
        print("[W] FFmpeg exit code not '0' but '{:d}'.".format(exit_code))
        return

    @abstractmethod
    def get_arguments(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def remove_temporary_files(self) -> None:
        raise NotImplementedError


class FFmpegUnifyFinalizer(FFmpegFinalizer):
    def get_arguments(self) -> list[str]:
        return [
            "-loglevel",
            "fatal",
            "-y",
            "-i",
            str(self.save_path.video),
            "-i",
            str(self.save_path.audio),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(self.save_path.final),
        ]

    def remove_temporary_files(self) -> None:
        self.save_path.video.unlink()
        self.save_path.audio.unlink()


class FFmpegCopyFinalizer(FFmpegFinalizer):
    def get_arguments(self) -> list[str]:
        return [
            "-loglevel",
            "fatal",
            "-y",
            "-i",
            str(self.save_path.video),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(self.save_path.final),
        ]

    def remove_temporary_files(self) -> None:
        self.save_path.video.unlink()


class SavePathFactory:
    def __init__(self, destination_directory: Path) -> None:
        self.destination_directory = destination_directory

    def create(self, video: Media) -> SavePath:
        final_file_name = video.get_file_name_mp4()
        return SavePath(
            self.destination_directory / final_file_name,
            self.destination_directory / final_file_name.replace(".mp4", ".video.mp4"),
            self.destination_directory / final_file_name.replace(".mp4", ".audio.mp4"),
        )


class FFmpegVideoDownloader:
    def __init__(
        self,
        statistics_collector: FFmpegFeedReelItemsCollector,
        index: int,
        video: Media,
        save_path: SavePath,
    ) -> None:
        self.statistics_collector = statistics_collector
        self.index = index
        self.video = video
        self.save_path = save_path
        if self.save_path.final.exists():
            print(f"[I] Story already exists: {self.save_path.final.name}")
            return
        print(
            f"[I] ({index + 1}/{len(self.statistics_collector.list_video_v)}) "
            f"Downloading video: {self.save_path.final.name}".format(),
        )

    def download_each_streams(self) -> bool:
        download_file(self.video.url, self.save_path.video)
        if self.statistics_collector.list_video_a[self.index] == "noaudio":
            return False
        download_file(self.statistics_collector.list_video_a[self.index], self.save_path.audio)
        return True


TypeVarAbstractFeedReelItemsCollector = TypeVar(
    "TypeVarAbstractFeedReelItemsCollector",
    bound=AbstractFeedReelItemsCollector,
)


class AbstractNewStatisticsCollector(Generic[TypeVarAbstractFeedReelItemsCollector]):
    def __init__(
        self,
        destination_directory: Path,
        statistics_collector: TypeVarAbstractFeedReelItemsCollector,
        *,
        taken_at: bool,
    ) -> None:
        self.destination_directory = destination_directory
        self.statistics_collector = statistics_collector
        self.taken_at = taken_at
        self.list_video_new: list[Path] = []
        self.list_image_new: list[Path] = []

    def collect_image(self, index: int, image: Media) -> None:
        final_file_name = image.get_file_name_jpeg()
        save_path = self.destination_directory / final_file_name
        if save_path.exists():
            print("[I] Story already exists: {:s}".format(final_file_name))
            return
        print(
            "[I] ({:d}/{:d}) Downloading image: {:s}".format(
                index + 1,
                len(self.statistics_collector.list_image),
                final_file_name,
            ),
        )
        try:
            download_file(image.url, save_path)
            self.list_image_new.append(save_path)
        except Exception as e:
            print("[W] An error occurred while iterating image stories: " + str(e))
            exit(1)

    def report(self) -> None:
        print_line()
        if len(self.list_image_new) == 0 and len(self.list_video_new) != 0:
            print("[I] No new stories were downloaded.")
            return
        print(
            "[I] Story downloading ended with "
            + str(len(self.list_image_new))
            + " new images and "
            + str(len(self.list_video_new))
            + " new videos downloaded.",
        )

    @abstractmethod
    def collect_video(self, index: int, video: Media) -> None:
        raise NotImplementedError


class FFmpegNewStatisticsCollector(AbstractNewStatisticsCollector[FFmpegFeedReelItemsCollector]):
    def collect_video(self, index: int, video: Media) -> None:
        save_path = SavePathFactory(self.destination_directory).create(video)
        downloader = FFmpegVideoDownloader(self.statistics_collector, index, video, save_path)
        try:
            has_audio = downloader.download_each_streams()
            self.create_finalizer(save_path, has_audio=has_audio).finalize()
            self.list_video_new.append(save_path.final)
        except Exception as e:
            print("[W] An error occurred while iterating HQ video stories: " + str(e))
            exit(1)

    def create_finalizer(self, save_path: SavePath, *, has_audio: bool) -> FFmpegFinalizer:
        if has_audio:
            return FFmpegUnifyFinalizer(save_path)
        return FFmpegCopyFinalizer(save_path)


class NewStatisticsCollector(AbstractNewStatisticsCollector[FeedReelItemsCollector]):
    def collect_video(self, index: int, video: Media) -> None:
        save_path = SavePathFactory(self.destination_directory).create(video)
        if save_path.final.exists():
            print("[I] Story already exists: {:s}".format(save_path.final.name))
            return
        print(
            "[I] ({:d}/{:d}) Downloading video: {:s}".format(
                index + 1,
                len(self.statistics_collector.list_video),
                save_path.final.name,
            ),
        )
        try:
            download_file(video.url, save_path.final)
            self.list_video_new.append(save_path.final)
        except Exception as e:
            print("[W] An error occurred while iterating video stories: " + str(e))
            exit(1)


TypeVarAbstractNewStatisticsCollector = TypeVar(
    "TypeVarAbstractNewStatisticsCollector",
    bound=AbstractNewStatisticsCollector[AbstractFeedReelItemsCollector],
)


class AbstractStoryMediaDownloader(
    Generic[TypeVarAbstractFeedReelItemsCollector, TypeVarAbstractNewStatisticsCollector],
):
    def __init__(
        self,
        destination_directory: Path,
        user: User,
        ig_client: ChallengeableClient,
        feed_reel_items_collector: TypeVarAbstractFeedReelItemsCollector,
        *,
        taken_at: bool = False,
        no_video_thumbs: bool = False,
    ) -> None:
        self.destination_directory = destination_directory
        self.user = user
        self.ig_client = ig_client
        self.feed_reel_items_collector = feed_reel_items_collector
        self.taken_at = taken_at
        self.no_video_thumbs = no_video_thumbs

    def execute(self) -> None:
        try:
            self._execute()
        except Exception as e:
            print("[E] A general error occurred: " + str(e))
            exit(1)
        except KeyboardInterrupt:
            print("[I] User aborted download.")
            exit(1)

    def _execute(self) -> None:
        try:
            feed = self.ig_client.user_story_feed(self.user.id)
        except Exception as e:
            print("[W] An error occurred trying to get user feed: " + str(e))
            return
        try:
            feed_json = feed["reel"]["items"]
            (Path(".pyinstastories_credentials") / "feed.json").write_text(json.dumps(feed_json), encoding="utf-8")
        except TypeError:
            print("[I] There are no recent stories to process for this user.")
            return

        for media in feed_json:
            self.feed_reel_items_collector.collect(media)

        new_statistics_collector = self.create_new_statistics_collector(self.feed_reel_items_collector)
        self.collect_mp4(self.feed_reel_items_collector, new_statistics_collector)

        print_line()
        print(
            "[I] Downloading image stories. ({:d} stories detected)".format(
                len(self.feed_reel_items_collector.list_image),
            ),
        )
        print_line()
        for index, image in enumerate(self.feed_reel_items_collector.list_image):
            new_statistics_collector.collect_image(index, image)
            # Prevent HTTP 429 error
            time.sleep(ChallengeableClient.REQUEST_INTERVAL)

        new_statistics_collector.report()

    @abstractmethod
    def create_new_statistics_collector(
        self,
        feed_reel_items_collector: TypeVarAbstractFeedReelItemsCollector,
    ) -> TypeVarAbstractNewStatisticsCollector:
        raise NotImplementedError

    @abstractmethod
    def collect_mp4(
        self,
        feed_reel_items_collector: TypeVarAbstractFeedReelItemsCollector,
        new_statistics_collector: TypeVarAbstractNewStatisticsCollector,
    ) -> None:
        raise NotImplementedError


class FFmpegStoryMediaDownloader(
    AbstractStoryMediaDownloader[FFmpegFeedReelItemsCollector, FFmpegNewStatisticsCollector],
):
    def create_new_statistics_collector(
        self,
        feed_reel_items_collector: FFmpegFeedReelItemsCollector,
    ) -> FFmpegNewStatisticsCollector:
        return FFmpegNewStatisticsCollector(
            self.destination_directory,
            feed_reel_items_collector,
            taken_at=self.taken_at,
        )

    def collect_mp4(
        self,
        feed_reel_items_collector: FFmpegFeedReelItemsCollector,
        new_statistics_collector: FFmpegNewStatisticsCollector,
    ) -> None:
        print(
            "[I] Downloading video stories. ({:d} stories detected)".format(
                len(feed_reel_items_collector.list_video_v),
            ),
        )
        print_line()
        for index, video in enumerate(feed_reel_items_collector.list_video_v):
            new_statistics_collector.collect_video(index, video)
            # Prevent HTTP 429 error
            time.sleep(ChallengeableClient.REQUEST_INTERVAL)


class StoryMediaDownloader(AbstractStoryMediaDownloader[FeedReelItemsCollector, NewStatisticsCollector]):
    def create_new_statistics_collector(
        self,
        feed_reel_items_collector: FeedReelItemsCollector,
    ) -> NewStatisticsCollector:
        return NewStatisticsCollector(
            self.destination_directory,
            feed_reel_items_collector,
            taken_at=self.taken_at,
        )

    def collect_mp4(
        self,
        feed_reel_items_collector: FeedReelItemsCollector,
        new_statistics_collector: NewStatisticsCollector,
    ) -> None:
        print(
            "[I] Downloading video stories. ({:d} stories detected)".format(
                len(feed_reel_items_collector.list_video),
            ),
        )
        print_line()
        for index, video in enumerate(feed_reel_items_collector.list_video):
            new_statistics_collector.collect_video(index, video)
            # Prevent HTTP 429 error
            time.sleep(ChallengeableClient.REQUEST_INTERVAL)


class DownloadDestinationDirectory:
    def __init__(self, output: Path | None) -> None:
        self.path = self.initialize_path(output) / "stories"
        if not self.path.is_dir():
            self.path.mkdir()
        print(f"[I] Files will be downloaded to {self.path}")
        print_line()

    def initialize_path(self, output: Path | None) -> Path:
        if not output:
            return Path.cwd()
        if output.is_dir():
            return output
        else:
            print("[W] Destination '{:s}' is invalid, falling back to default location.".format(output))
            return Path.cwd()

    def ensure_directory_for_user(self, username: str) -> Path | None:
        path = self.path / username
        try:
            if not path.is_dir():
                path.mkdir()
            return path
        except Exception as e:
            print(str(e))
            return None


def restart_script() -> None:
    """Restarts the current program, with file objects and descriptors cleanup."""
    try:
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.net_connections():
            os.close(handler.fd)
    except Exception as e:
        logging.error(e)
    python = sys.executable
    os.execl(python, python, *sys.argv)


class RetriableDownloader:
    LOOP_TIME = 3

    def __init__(
        self,
        ig_client: ChallengeableClient,
        commandline: Commandline,
        user: User,
        destination_directory: Path,
    ) -> None:
        self.ig_client = ig_client
        self.commandline = commandline
        self.user = user
        self.destination_directory = destination_directory

    def try_to_download(self) -> None:
        for time_loop in range(self.LOOP_TIME):
            self.download(time_loop)

    def download(self, time_loop: int) -> None:
        try:
            return self._download()
        except Exception as e:
            print("[E] ({:d}) Download failed: {:s}.".format(time_loop + 1, str(e)))
            args = self.commandline.args
            if str(e) == "login_required" and (args.username and args.password):
                print("[W] Trying to re-login...")
                self.ig_client = LoginMethodSelector(args.username, args.password, force_login=True).login()
                print("[W] Restart Script...")
                restart_script()
        if time_loop + 1 >= self.LOOP_TIME:
            print("[E] Retry failed three times, skipping user.")
            return print_line()
        print("[W] Trying again in 5 seconds.")
        time.sleep(5)
        print_line()

    def _download(self) -> None:
        follow_res = self.ig_client.friendships_show(self.user.id)
        if follow_res.get("is_private") and not follow_res.get("following"):
            raise Exception("You are not following this private user.")
        args = self.commandline.args
        if self.commandline.hq_videos:
            story_media_downloader = self.create_ffmepeg_story_media_downloader(args)
        else:
            story_media_downloader = self.create_story_media_downloader(args)
        story_media_downloader.execute()

    def create_ffmepeg_story_media_downloader(self, args: type[MyProgramArgs]) -> AbstractStoryMediaDownloader:
        feed_reel_items_collector = FFmpegFeedReelItemsCollector(
            taken_at=args.takenat,
            no_video_thumbs=args.novideothumbs,
        )
        return FFmpegStoryMediaDownloader(
            self.destination_directory,
            self.user,
            self.ig_client,
            feed_reel_items_collector,
            taken_at=args.takenat,
            no_video_thumbs=args.novideothumbs,
        )

    def create_story_media_downloader(self, args: type[MyProgramArgs]) -> AbstractStoryMediaDownloader:
        feed_reel_items_collector = FeedReelItemsCollector(
            taken_at=args.takenat,
            no_video_thumbs=args.novideothumbs,
        )
        return StoryMediaDownloader(
            self.destination_directory,
            self.user,
            self.ig_client,
            feed_reel_items_collector,
            taken_at=args.takenat,
            no_video_thumbs=args.novideothumbs,
        )


class InstagramStoriesDownloader:
    def __init__(self) -> None:
        print_line()
        print(
            "[I] PYINSTASTORIES (SCRIPT V{:s} - PYTHON V{:s}) - {:s}".format(
                script_version,
                python_version,
                time.strftime("%Y-%m-%d %I:%M:%S %p"),
            ),
        )
        print_line()
        self.commandline = Commandline()
        self.ig_client = login_instagram(self.commandline)
        self.user_factory = UserFactory(self.ig_client)
        print_line()
        self.download_destination_directory = DownloadDestinationDirectory(self.commandline.args.output)

    def download_stories(self) -> None:
        try:
            self._download_stories()
        except KeyboardInterrupt:
            print_line()
            print("[I] The operation was aborted.")
            print_line()
        exit(0)

    def _download_stories(self) -> None:
        usernames = self.commandline.usernames
        for index, username in enumerate(usernames):
            self.download_each_users_stories(index, username)

    def download_each_users_stories(self, index: int, username: str) -> None:
        user = self.user_factory.create(username)
        print("[I] Getting stories for: {:s}".format(user.name))
        print_line()
        destination_directory_for_user = self.download_destination_directory.ensure_directory_for_user(user.name)
        if not destination_directory_for_user:
            print("[E] Could not make required directories. Please create a 'stories' folder manually.")
            exit(1)
        RetriableDownloader(self.ig_client, self.commandline, user, destination_directory_for_user).try_to_download()
        usernames = self.commandline.usernames
        if (index + 1) != len(usernames):
            print_line()
            print(f"[I] ({index + 1}/{usernames}) 5 second time-out until next user...")
            time.sleep(5)
        print_line()


def try_to_run() -> None:
    try:
        InstagramStoriesDownloader().download_stories()
    # Reason: To return a non-zero exit code
    except BaseException as e:  # noqa: E722,H201,RUF100  pylint: disable=bare-except
        print(f"[E] Unexpected Exception: {e}")
        traceback.print_tb(e.__traceback__, file=sys.stdout)
        print_line()
        sys.exit(99)


def main() -> None:
    try_to_run()


if __name__ == "__main__":
    main()
