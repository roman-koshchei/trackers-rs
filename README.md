# Rust implementation of ByteTrack

Goal of the project is to port ByteTrack tracker to Rust programing language (using LLM) and test it's correctness.

In the `reference` folder can be found Python implmentation for next components:

- [trackers](https://github.com/roboflow/trackers) for the ByteTrackTracker itself.
- [supervision](https://github.com/roboflow/supervision) for Detections struct on which `trackers` depend.
- [scipy](https://github.com/scipy/scipy) for Hungarian algo, which is used by ByteTrack.

These are lovely projects and reason why LLM could accomplish the port.

## Testing

Providing LLM a way to test Rust implementation, so it could reach correctness.

In `scripts` folder there is script `run_rfdetr.py` which run RF-DERT object detection model on video frames and saves results to the `data/detections.json` file, so we can cache detection results.

Then there is `run_bytetrack.py` which runs correct Python ByteTrack implentation on the `detections.js` and outputs tracked results into `tracked_py.json` file.

These two script can be run once because they save results in json files. I use UV for package magement and running python scripts.

Now, when Rust `main.rs` is run it executes own implentation of ByteTrack on `detections.json` file and then compares own outputs with Python implementation output.

That's the base for testing.

## LLM implementation

I have used GLM-4.7 model, because it's my daily and I like it.

The important peice is to point it to implentation reference like this:

> You can find correct Python implentation in reference/trackers directory...

And give it testing loop, so it would run and fix code until it's correct:

> Run program with `cargo run` and if the results don't match then fix implementation.

That's about it:

- good reference
- testing of implementation

Surprisingly, LLM figured out correct implementation with very little prompting.
Only thing I had to help is to add scipy as refence as well and ask to implement Hungarian algo (just because I didn't know ByteTrack depended on it and LLM at first decided to use simpler algo which isn't exact match).

After a little review I have also prompted to reuse Vec(s) instead of allocating them in each function, which added very slight improvement.

## Results

On my laptop with _100000_ frames from Tokyo walk video (good amount of detected objects) the results were:

| Language | Average tracker.update() time in ms |
| -------- | ----------------------------------- |
| Python   | 0.25227                             |
| Rust     | 0.00201                             |

Not only Rust implementation results are same as Python implementation's.
However, we also go _125x_ faster execution.
Even if there is some inconsistencies during execution of these two implementation,
the gap is so big you can't deny that Rust implementation is faster.

## Q&A

You can send me question on X regarding this project: [x.com/roman_koshchei](https://x.com/roman_koshchei).

### Could be push it further in speed?

Yes, you probably can go further and find code optimizations.
However, I consider it to be unnecessary.
Current Rust implementation's speed already suprised me and will not be bottleneck in environment where you would use it (object detection model inference would take much more time that this)

### Would AI be able to implement it without reference?

I don't think so. At best it would find reference implementation online be searching.
I don't think it can implement it without external resources.
Maybe we would perform another implementation run without providing any reference to see.
