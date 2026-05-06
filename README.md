# Object trackers implementation in Rust

The goal is to implement ByteTrack object tracker in Rust language based on Python implementation as a reference and point of correctness for tests.

Correctness test is important part of the project to ensure Rust implementation matches Python one.

As a reference we use next project which are added as git submodules for reference:

- [trackers](https://github.com/roboflow/trackers) for the ByteTrackTracker itself.
- [supervision](https://github.com/roboflow/supervision) for Detections struct on which `trackers` depend.
- [scipy](https://github.com/scipy/scipy) for Hungarian algo, which is used by ByteTrack.

## Testing

### Preparation

First we `uv run scripts/run_rfdetr.py` generic RF-DETR model to detect object on random town walk video I have found. Detections are saved in `data/detections.json` file for future runs of trackers on these detections and single point of comparision.

Then run `uv run scripts/run_bytetrack.py` which will run Python ByteTrack tracking on the `detections.json` and save results of tracking into `data/tracked_py.json`.

Now you have prepared setup for Rust implementation testing.

### Running Rust implementation

Rust version has tracking and comparison in `main.rs`. So build it in release mode:

```bash
cargo build --release
```

And run:

```bash
./target/release/trackers_rs.exe
```

The output will tell you whether Rust implementation matches Python one (obviously with some small error allowed because of floating point operations). As well it will measure time required to run Rust tracking and copare with Python (only tracker update function which is done in each loop iteration is measured).

## Performance

On my laptop I got around 50x improvement in speed in comparison with Python implementation on average.
It might be not the best benchmark, but we can surely tell that it matches and likely faster than Python implementation.

```txt
Performance comparison:
  Python avg: 0.2388 ms
  Rust avg: 0.0046 ms
  Rust is 51.80x faster than Python
```

## LLM usage

> Did we use LLMs during implementation?

Yes, we did. We used it to help us find reference points, what algorithms need to be implemented and so on.
It's great research helper.

Because of our testing setup (one of the crutial parts of project) we were also able to test out [autoresearch](https://github.com/karpathy/autoresearch) for this project. There is a benchmark with a number (avg tracking speed) and LLM is given ability to change code and run the benchmark to verify if changes improved the result or not. It was cool to see and there were improvements.
