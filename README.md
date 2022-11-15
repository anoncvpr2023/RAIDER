# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch

- Help: `python utils/main.py --help`

## Datasets
- Seq. CIFAR-100: `seq-cifar100-online`
- Seq. miniImageNet: `seq-miniimg-online`

## Models
- DualNet:  `dualnet`
- CoPE:     `cope`
- X-DER:    `xder`
- ER-ACE:   `er_ace`
- Joint:    `joint`
- Finetune: `sgd`

## Arguments

### Regularizations
- RAIDER: `<model name>_raider`
    - requires `--ptx_alpha` (`lambda_r`)
- CSSL (barlow twins): `<model name>_cssl`
    - requires `--inv_alpha` (`lambda_r`)

*Note: Not available for SGD and Joint*

### Mandatory arguments
- `--dataset`
- `--model`
- `--buffer_size`:  not available for `sgd` and `joint`
- `--lr`            (not available for `sgd` and `joint`)

**Other arguments depend on the base model, see --help**

## Ex: Run ER-ACE + RAIDER: example on seq-cifar100-online with buffer size 500

Command:

`python utils/main.py --dataset=seq-cifar100-online --model=er_ace_raider --buffer_size=500 --lr=0.1 --ptx_alpha=0.01`

