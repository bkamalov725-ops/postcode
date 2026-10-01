"""Publishing must not carry a private staging file's ACL into the user folder."""
from pathlib import Path
import tempfile
import unittest
from scripts.prepare_data import _publish


class PublishPermissionsTests(unittest.TestCase):
    def test_published_file_has_its_own_inode_and_identical_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/'stage'; stage.mkdir()
            target=Path(tmp)/'target'
            source=stage/'model.bin'; source.write_bytes(b'unchanged model weights')
            self.assertEqual(_publish(stage,target,['model.bin']),1)
            published=target/'model.bin'
            self.assertEqual(source.read_bytes(),published.read_bytes())
            self.assertNotEqual(source.stat().st_ino,published.stat().st_ino)
            self.assertEqual(list(target.glob('*.tmp')),[])


if __name__=='__main__':
    unittest.main()
