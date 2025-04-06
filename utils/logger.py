# -*- coding: utf-8 -*-
"""global logger config
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch


class SpecificLogFilter(logging.Filter):
    def filter(self, record):
        # Only log messages containing 'specific' in the message
        return 'timestamp' in record.getMessage()


def logger_config(log_savepath, logging_name):
    '''logger config
    '''
    # get logger name
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)

    # get file handler and set level
    file_handler = logging.FileHandler(log_savepath, encoding='UTF-8')
    file_handler.setLevel(logging.DEBUG)

    # Add the filter to the handler
    # log_filter = SpecificLogFilter()
    # file_handler.addFilter(log_filter)

    # format the file handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # console sream handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # add handler for logger objecter
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def log_inference_figure(ori_time_sig, pred_time_sig, metrics, save_dir=None):
    """show the estimated and ground truth signal, show the signal metric on the figure

    Parameters
    ----------
    ori_time_sig : np.array
        ground truth signal
    pred_time_sig : np.array
        estimated signal
    metric : dictionary
        metric infomation
    save_dir : string, optional
        image save directory, by default None (not save)
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ori_time_sig, c='b')
    ax.plot(pred_time_sig, c='r', alpha=0.8)
    ax.set_ylim(-np.max(np.abs(ori_time_sig))*1,
                np.max(np.abs(ori_time_sig))*1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    font_size = 26
    text_pos_x = 0.65
    text_pos_y = 0.10

    plt.text(text_pos_x, 0.50 - text_pos_y, f"Angle err: {metrics['Angle']:.2f}", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.44 - text_pos_y, f"Amp. err: {metrics['Amplitude']:.3f}", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.38 - text_pos_y, f"Env. err: {metrics['Envelope']:.3f}", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.32 - text_pos_y, f"T60 err: {metrics['T60'] * 100:.2f}%", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.26 - text_pos_y, f"C50 err: {metrics['C50']:.2f} db", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.20 - text_pos_y, f"EDT err: {metrics['EDT']:.3f} s", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)

    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.tight_layout()

    if save_dir is not None:
        # Change the path and file name as needed
        plt.savefig(save_dir, dpi=300, pad_inches=0)
    plt.close("all")

    return


def plot_and_save_figure(pred_sig, ori_sig, pred_time, ori_time, position_rx, position_tx, mode_set, save_path):
    plt.figure(1, figsize=(16, 12))
    plt.suptitle(f"{mode_set} set")
    plt.subplot(231)
    plt.title("Real")
    plt.plot(np.real(pred_sig.detach().cpu().numpy().flatten()))
    plt.plot(np.real(ori_sig.type(
        torch.complex64).flatten().cpu().numpy()), alpha=0.5)

    plt.subplot(234)
    plt.title("Imaginary")
    plt.plot(np.imag(pred_sig.detach().cpu().numpy().flatten()))
    plt.plot(np.imag(ori_sig.type(
        torch.complex64).flatten().cpu().numpy()), alpha=0.5)

    plt.subplot(232)
    plt.plot(pred_time.detach().cpu().numpy().flatten())
    plt.plot(ori_time.detach().cpu().numpy().flatten(), alpha=0.5)

    plt.subplot(235)
    plt.scatter(position_rx[0], position_rx[1], c='b')
    plt.scatter(position_tx[0], position_tx[1], c='r')
    plt.grid(True)
    plt.axis("equal")

    plt.subplot(233)
    plt.plot(np.abs(pred_sig.detach().cpu().numpy().flatten()))
    plt.plot(np.abs(ori_sig.type(torch.complex64).flatten().cpu().numpy()), alpha=0.5)
    plt.ylim(0)

    plt.subplot(236)
    plt.plot(np.angle(pred_sig.detach().cpu().numpy().flatten()))
    plt.plot(np.angle(ori_sig.type(
        torch.complex64).flatten().cpu().numpy()), alpha=0.5)
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close("all")


if __name__ == '__main__':
    logger = logging.getLogger('avr')
    file_handlers = [handler for handler in logger.handlers if isinstance(
        handler, logging.FileHandler)]
    file_handler = file_handlers[0]
    file_handler.setLevel(logging.DEBUG)
