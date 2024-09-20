To deploy model

```
modal deploy model.py
```

To test latency, update the URL in bench.py and run

```
pytest bench.py
```


------------------------

For just benchmarking modal latency

```
modal deploy test.py
```

```
pytest test.py
```