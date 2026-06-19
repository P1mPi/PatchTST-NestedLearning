from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

from utils.M3_optim import M3Optimizer

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        #model_optim = M3Optimizer(self.model.parameters(), lr=self.args.learning_rate, alpha=0.1)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    
    # Nuevo test para NL con Trigger Estadístico
    # Nuevo test para NL con Trigger Dinámico y soporte de Baseline inteligente
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        criterion = self._select_criterion()
        preds, trues, inputx = [], [], []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

  
        # Es estático si la política es 'none' o el learning rate de inferencia es 0
        is_static = (getattr(self.args, 'update_policy', 'none') == 'none') or (getattr(self.args, 'cms_lr', 0.0) == 0.0)

        # BASELINE
        if is_static:
            print(" Ejecutando Inferencia ESTÁTICA ")
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    outputs = self.model(batch_x)

                    f_dim     = -1 if self.args.features == 'MS' else 0
                    outputs   = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_s = batch_y[:, -self.args.pred_len:, f_dim:]

                    pred = outputs.detach().cpu().numpy()
                    true = batch_y_s.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.detach().cpu().numpy())

                    if i % 10 == 0:
                        inp  = batch_x.detach().cpu().numpy()
                        gt   = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                        pd_v = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd_v, os.path.join(folder_path, str(i) + '.pdf'))


        # NESTED LEARNING
        # Aquí entra 'cms', 'cms3' y también 'flatten_nl' si LR > 0
        else:
            target_name = 'mid_cms' if getattr(self.args, 'use_mid_cms', 0) == 1 else 'head'
            
            print(f" Ejecutando Inferencia DINÁMICA (Nested Learning)")
            print(f"  -> Política: {self.args.update_policy} | LR: {self.args.cms_lr} | Target: {target_name}")

            self.model.eval()
            # Congelamos los gradientes de todo excepto del target
            for name, param in self.model.named_parameters():
                if target_name in name and 'head_estatica' not in name:
                    param.requires_grad = True  # Fast weights (aprenden)
                else:
                    param.requires_grad = False # Slow weights (congelados)

            # Optimizador con el LR dinámico
            cms_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.cms_lr)

            torch.set_grad_enabled(True) 
            
            historial_losses = []
            window_size = 30 # Tamaño de la ventana para el cálculo del umbral dinámico
            
            veces_actualizado = 0
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x   = batch_x.float().to(self.device)
                batch_y   = batch_y.float().to(self.device)

                # Si el target es 'head', desconectamos el backbone del grafo (cms_mode=True) para ahorrar RAM
                # Si el target es 'mid_cms', el gradiente DEBE atravesar el backbone, así que no se desconecta (cms_mode=False)
                do_detach = (target_name == 'head') 
                outputs = self.model(batch_x, cms_mode=do_detach)

                f_dim     = -1 if self.args.features == 'MS' else 0
                outputs   = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_s = batch_y[:, -self.args.pred_len:, f_dim:]

                # Error escalar del batch actual
                loss = criterion(outputs, batch_y_s) 

                # CONTROL DEL TRIGGER DE ACTUALIZACIÓN
                actualizar = False
                policy = self.args.update_policy

                if policy == 'always':
                    actualizar = True
                elif policy == '5steps':
                    if i % 5 == 0:
                        actualizar = True
                elif policy == 'spc':
                    # Si tenemos suficientes datos históricos para calcular una media y std fiables
                    if len(historial_losses) > 5: 
                        # Convertimos a tensor para usar matemáticas rápidas
                        hist_tensor = torch.tensor(historial_losses)
                        mu = hist_tensor.mean()
                        sigma = hist_tensor.std()
                        
                        # Umbral estadístico real: Media histórica + 2 desviaciones
                        umbral = mu + (1.0 * sigma)
                        
                        if loss.item() > umbral.item():
                            actualizar = True
                            # Opcional: Descomenta esto para ver CÚANDO se dispara
                            # print(f"Batch {i}: Drift detectado! Loss: {loss.item():.4f} > {umbral.item():.4f}")

                # Backward y Step sobre el target
                if actualizar:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    cms_optim.step()
                    veces_actualizado += 1

                # Actualizamos la memoria histórica (DESPUÉS del trigger para no auto-sesgarlo)
                # OJO: Guardamos el error del modelo ANTES de actualizar, para medir bien la degradación
                historial_losses.append(loss.item())
                if len(historial_losses) > window_size:
                    historial_losses.pop(0) # Mantenemos solo los últimos 'window_size' elementos

                # Guardar predicciones
                pred = outputs.detach().cpu().numpy()
                true = batch_y_s.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                # Guardar visualizaciones
                if i % 10 == 0:
                    inp  = batch_x.detach().cpu().numpy()
                    gt   = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd_v = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd_v, os.path.join(folder_path, str(i) + '.pdf'))

            torch.set_grad_enabled(False) 
            print(f"\nRESUMEN: El CMS se actualizó {veces_actualizado} veces de {len(test_loader)} posibles.\n")


        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds  = np.array(preds).reshape(-1, preds[0].shape[-2],  preds[0].shape[-1])
        trues  = np.array(trues).reshape(-1, trues[0].shape[-2],  trues[0].shape[-1])
        inputx = np.array(inputx).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        # Escribir en result.txt
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        return
    

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
