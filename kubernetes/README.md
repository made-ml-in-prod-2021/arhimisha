# Homework 4. Kubernetes

Проект подготовлен для испольования в kubernetes кластере [google cloud](https://cloud.google.com/kubernetes-engine).

Для работы с кластером необходимо установить:
 - Google Cloud SDK - [https://cloud.google.com/sdk/docs/install?hl=ru](https://cloud.google.com/sdk/docs/install?hl=ru)
 - kubectl - [https://kubernetes.io/docs/tasks/tools/](https://kubernetes.io/docs/tasks/tools/)
 - Lens (опционально) - [https://k8slens.dev/](https://k8slens.dev/)

kubectl должен быть совместимой с google cloud кластером версии.

После настройки кластера и установки приложений необходимо подключить утилиту kubectl к кластеру. 
Для это необходимо скопировать команду подключения, доступную по кнопке с троеточием в таблице кластеров в консоле google cloud.

Для запуска pod необходимо выполнить команду:
```
kubectl apply -f file_name.yaml
```
После запуска можно проверить результат с помощью Lens или с помощью команды:
```
kubectl describe pod/<pod_name>
```
Для осуществления запросов к запущенному контейнеру можно воспользоваться командой:
```
kubectl port-forward pod/<pod_name> 80:80
```

Реализованны следующие манифесты:
* [online-inference-pod.yaml](./online-inference-pod.yaml) простой Pod;
* [online-inference-pod-resources.yaml](./online-inference-pod-resources.yaml) простой Pod с указанием ресурсов;
* [online-inference-pod-probes.yam](./online-inference-pod-probes.yaml) Pod с эмуляцией долгого запуска приложения и последующего падения;
* [online-inference-replicaset.yaml](./online-inference-replicaset.yaml) ReplicaSet с запуском трех Pod;
* [online-inference-deployment-blue-green.yaml](./online-inference-deployment-blue-green.yaml) Deployment приложения со cтратегией обновления с предварительным запуском всех Pod с новой версией и последующим отключением со старой;
* [online-inference-deployment-rolling-update.yaml](./online-inference-deployment-rolling-update.yaml) Deployment приложение со cтратегией постепенного обновления Pod.
