for min_freq in 1 5 10 15 20; do
	for max_len in 20 25 30 35 40; do
		for embed_dim in 50 100 150; do
			for hidden_dim in 50 100 150; do
				for lr in 1e-2 1e-3 1e-4; do
					for train_bz in 32 64 128; do
						python main.py \
							--min-freq ${min_freq} \
							--max-len ${max_len} \
							--embed-dim ${embed_dim} \
							--hidden-dim ${hidden_dim} \
							--lr ${lr} \
							--train-bz ${train_bz}
					done
				done
			done
		done
	done
done