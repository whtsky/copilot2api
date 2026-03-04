FROM golang:1.26-alpine AS builder
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /copilot2api .

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /copilot2api /copilot2api
ENV HOME=/root
ENV COPILOT2API_HOST=0.0.0.0
ENV COPILOT2API_PORT=7777
EXPOSE 7777
ENTRYPOINT ["/copilot2api"]
