from unittest.mock import Mock

import pytest

from marovi.modules.wiki import WikiClient


def _mock_response(json_data):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json = Mock(return_value=json_data)
    return resp


def test_wiki_client_download_page():
    client = WikiClient("https://wiki.example/api.php")
    client.session = Mock()
    client.session.get.return_value = _mock_response(
        {
            "query": {
                "pages": {
                    "1": {
                        "revisions": [
                            {"slots": {"main": {"*": "content"}}}
                        ]
                    }
                }
            }
        }
    )
    text = client.download_page("Title")
    assert text == "content"
    client.session.get.assert_called()


def test_wiki_client_upload_file(tmp_path):
    client = WikiClient("https://wiki.example/api.php")
    client.csrf_token = "CSRF"
    client.session = Mock()
    client.session.post.return_value = _mock_response({"upload": {"result": "Success"}})
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")
    client.upload_file("file.txt", str(file_path))
    assert client.session.post.called
    call_args = client.session.post.call_args
    assert call_args.kwargs["data"]["action"] == "upload"
