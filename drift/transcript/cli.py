"""Explicit capture, MCDMA receive and reader commands; no API interception or remote launch."""
from pathlib import Path

from drift.transcript.files import read
from drift.transcript.importer import from_chat_messages, strict_json
from drift.transcript.schema import Transcript


def add_commands(commands):
    group = commands.add_parser('transcript', help='build local memory from an explicitly supplied API transcript')
    sub = group.add_subparsers(dest='transcript_command', required=True)
    importer = sub.add_parser('import', help='convert a completed text-only chat messages array')
    importer.add_argument('--input', type=Path, required=True)
    importer.add_argument('--output', type=Path, required=True)
    for name in ('provider', 'model', 'conversation-id'):
        importer.add_argument('--' + name, required=True)
    validate = sub.add_parser('validate', help='validate a bounded transcript without loading a model')
    capture = sub.add_parser('capture', help='prefill the owned local GLM and export only latent tensors')
    for parser in (validate, capture):
        parser.add_argument('--input', type=Path, required=True)
    for name in ('tokenizer', 'export-root', 'output'):
        capture.add_argument('--' + name, type=Path, required=True)
    capture.add_argument('--tokenizer-sha256', required=True)
    capture.add_argument('--model', required=True)
    capture.add_argument('--arena', type=Path)
    capture.add_argument('--url', default='http://127.0.0.1:8888/v1/completions')
    capture.add_argument('--max-tokens', type=int, default=4096)
    receive = sub.add_parser('receive', help='pull a hash-pinned tensor-only artifact through handoffd')
    for name in ('peer', 'remote', 'sha256'):
        receive.add_argument('--' + name, required=True)
    receive.add_argument('--size', type=int, required=True)
    receive.add_argument('--socket-path', default='/tmp/handoffd.sock')
    receive.add_argument('--output', type=Path, required=True)
    answer = sub.add_parser('answer', help='answer from a verified memory using the local Qwen oMLX runtime')
    for name in ('memory', 'question', 'output', 'checkpoint', 'manifest', 'index'):
        answer.add_argument('--' + name, type=Path, required=True)
    for name in ('sha256', 'manifest-sha256', 'index-sha256'):
        answer.add_argument('--' + name, required=True)
    answer.add_argument('--need-gb', type=float, required=True)
    answer.add_argument('--max-new', type=int, default=64)
    answer.add_argument('--max-rows', type=int, default=8192)
    for parser in (capture, receive, answer):
        parser.add_argument('--timeout', type=float, required=True, help='explicit wall-time budget, at most 600 seconds')


def dispatch(args):
    if args.transcript_command in ('capture', 'answer'):
        from drift.transcript.process import bounded
        return bounded(args)
    return execute(args)


def execute(args):
    operation = args.transcript_command
    if operation == 'import':
        from drift.transcript.files import publish
        transcript = from_chat_messages(strict_json(read(args.input, 1048576)), provider=args.provider,
                                        model=args.model, conversation_id=args.conversation_id)
        publish(args.output, (transcript.canonical + '\n').encode())
        return {'status': 'PASSED', 'scope': 'import_only', **transcript.receipt()}
    if operation in ('validate', 'capture'):
        transcript = Transcript.parse(strict_json(read(args.input, 1048576)))
        if operation == 'validate':
            return {'status': 'PASSED', 'scope': 'schema_only', **transcript.receipt()}
        from drift.transcript.capture import capture
        return capture(transcript, **{key: getattr(args, key) for key in (
            'tokenizer', 'tokenizer_sha256', 'model', 'export_root', 'output', 'url', 'timeout', 'max_tokens', 'arena')})
    if operation == 'receive':
        from drift.transcript.receive import receive
        return receive(**{key: getattr(args, key) for key in (
            'peer', 'remote', 'sha256', 'size', 'output', 'socket_path', 'timeout')})
    if operation == 'answer':
        from drift.transcript.answer import answer
        from drift.transcript.qwen import QwenReader
        def factory():
            return QwenReader(**{key: getattr(args, key) for key in (
                'checkpoint', 'manifest', 'manifest_sha256', 'index', 'index_sha256', 'need_gb')})
        return answer(reader_factory=factory, **{key: getattr(args, key) for key in (
            'memory', 'sha256', 'question', 'output', 'max_new', 'max_rows', 'timeout')})
    raise ValueError('TRANSCRIPT_COMMAND')
