"""Rate-limit regression tests; all API responses and clocks are simulated."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from .solver_code_embeddings import RequestLimiter, generate, request_embeddings, retry_delay


class Response(io.StringIO):
    headers = {'x-request-id': 'unit-test-only'}


def rate_error(detail='rate limit', headers=None):
    return HTTPError('https://api.voyageai.com', 429, 'not-logged', headers or {},
                     io.BytesIO(json.dumps({'detail': detail}).encode()))


def success():
    return Response(json.dumps({'model': 'voyage-code-4',
                                'data': [{'index': 0, 'embedding': [1.] * 1024}]}))


class RequestTests(unittest.TestCase):
    def test_retry_after_and_redacted_service_detail(self):
        error = rate_error('3 RPM, 10,000 TPM; do not log private-value', {'Retry-After': '2'})
        output = io.StringIO()
        with patch('urllib.request.urlopen', side_effect=[error, success()]) as request:
            with patch('time.sleep') as sleep, contextlib.redirect_stdout(output):
                vectors, _ = request_embeddings(['def a(): pass'], 'unit-test-key', token_count=100)
        self.assertEqual(vectors.shape, (1, 1024))
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(2.)
        self.assertNotIn('private-value', output.getvalue())
        self.assertNotIn('unit-test-key', output.getvalue())

    def test_unknown_rate_limit_waits_a_full_minute(self):
        self.assertEqual(retry_delay(rate_error(), 0), 61.)
        deadline = {'Retry-After': 'Thu, 08 Oct 2026 00:01:00 GMT'}
        with patch('time.time', return_value=1791417600.):
            self.assertEqual(retry_delay(rate_error(headers=deadline), 0), 60.)

    def test_quota_and_impossible_token_batch_fail_without_retry(self):
        for detail, tokens, message in [('insufficient_quota; private-value', 100, 'quota/payment'),
                                        ('Rate limit 3 RPM, 10000 TPM', 22004, '22004 tokens'),
                                        ('Rate limit 0 RPM', 100, 'zero rate limit')]:
            with self.subTest(detail=detail), patch('urllib.request.urlopen', side_effect=rate_error(detail)) as request:
                with patch('time.sleep') as sleep, self.assertRaisesRegex(RuntimeError, message) as raised:
                    request_embeddings(['def a(): pass'], 'unit-test-key', token_count=tokens)
                self.assertEqual(request.call_count, 1)
                sleep.assert_not_called()
                self.assertNotIn('private-value', str(raised.exception))

    def test_terminal_rate_limit_reports_limits_without_remote_body(self):
        errors = [rate_error('3 RPM, 10000 TPM; private-value') for _ in range(5)]
        with patch('urllib.request.urlopen', side_effect=errors), patch('time.sleep') as sleep:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'RPM=3, TPM=10000') as raised:
                request_embeddings(['def a(): pass'], 'unit-test-key', token_count=100)
        self.assertEqual(sleep.call_count, 4)
        self.assertNotIn('private-value', str(raised.exception))

    def test_rolling_request_and_token_limits(self):
        for rpm, tpm, sizes in [(3, 10000, [6000, 4000, 1000]), (1, 100000, [1, 1])]:
            clock = [0.]
            limiter = RequestLimiter(rpm, tpm)
            def advance(seconds):
                clock[0] += seconds
            with patch('time.monotonic', side_effect=lambda: clock[0]), patch('time.sleep', side_effect=advance) as sleep:
                with contextlib.redirect_stdout(io.StringIO()):
                    for size in sizes:
                        limiter.wait(size)
            sleep.assert_called_once_with(61.)
        with self.assertRaises(ValueError):
            RequestLimiter(0, 10000)
        with self.assertRaises(ValueError):
            RequestLimiter(3, 100).wait(101)

    def test_token_bounded_batches_and_resume_only_uncached_chunks(self):
        # Minimal cache-only fixture, never used to assemble production vectors.
        chunks = [dict(id=str(i), tokens=3000, text=f'def f{i}(): pass', text_sha256=str(i)) for i in range(3)]
        corpus = dict(chunks=chunks, ready=True, preprocessing='test', tokenizer_sha256='test',
                      max_tokens=4096, manifest_sha256='test')
        def response(texts, key, **kwargs):
            return np.ones((len(texts), 1024), dtype=np.float32), dict(source='voyage-http', timestamp_utc='unit-test-only')
        with tempfile.TemporaryDirectory() as temp, patch('code.V4.solver_code_embeddings.validate_corpus'):
            cache = Path(temp)
            with patch('code.V4.solver_code_embeddings.api_key', return_value='unit-test-key'):
                with patch('code.V4.solver_code_embeddings.request_embeddings', side_effect=response) as request:
                    with contextlib.redirect_stdout(io.StringIO()):
                        generate(corpus, cache, True, 10000, batch_size=4, tokens_per_minute=5000)
            self.assertEqual(request.call_count, 3)
            self.assertTrue(all(call.kwargs['token_count'] == 3000 for call in request.call_args_list))
            with patch('code.V4.solver_code_embeddings.api_key') as key, patch('urllib.request.urlopen') as network:
                result = generate(corpus, cache, True, 0)
            self.assertEqual(result['uncached_chunks'], 0)
            key.assert_not_called()
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
