import unittest

from flashengine.protocol import parse_output
from flashengine.streaming import HermesTextStream


class HermesStreamingTest(unittest.TestCase):
    def check_splits(self, text, expected_prefix):
        for split in range(len(text) + 1):
            parser = HermesTextStream()
            first = parser.feed(text[:split])
            second = parser.feed(text[split:])
            self.assertEqual(first + second, expected_prefix, (split, text))
            self.assertEqual(parser.sent, expected_prefix)

    def test_plain_text_streams_across_boundaries(self):
        text = '你好，world。再见！'
        self.check_splits(text, text)
        parser = HermesTextStream()
        chunks = [parser.feed(char) for char in text]
        self.assertTrue(any(chunks[:-1]))
        self.assertEqual(''.join(chunks), text)

    def test_json_and_xml_tool_markers_never_leak(self):
        for body in ('<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                     '<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>'):
            text = '正在检查。 ' + body
            self.check_splits(text, '正在检查。')
            parsed = parse_output(text, 'hermes', tools=[{'function': {'name': 'bash'}}])
            self.assertEqual(len(parsed.tool_calls), 1)
            self.assertEqual(parsed.tool_calls[0]['function']['name'], 'bash')

    def test_incomplete_marker_and_think_are_withheld(self):
        for marker in ('<tool_call>', '<function=bash>', '<think>', '<|im_end|>'):
            self.check_splits('正文' + marker + 'secret', '正文')
        parsed = parse_output('<tool_call><function=broken>', 'hermes', tools=[])
        self.assertEqual(parsed.tool_calls, [])

    def test_whitespace_and_possible_marker_prefix(self):
        parser = HermesTextStream()
        emitted = ''.join(parser.feed(x) for x in ('  hello ', '<', 'x', '> world '))
        self.assertEqual(emitted, 'hello <x> world')
        self.assertEqual(parser.sent, 'hello <x> world')


if __name__ == '__main__':
    unittest.main()
