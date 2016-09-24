/**
 * Created by Gabriel Alacchi on 9/24/16.
 */

var tcp = require('net');
var cli = require('./lib/aud-cli');

var server = tcp.createServer(function listener(socket) {
  console.log("New socket connection from: " + socket.remoteAddress);

  socket.on('data', function(chunk) {
    console.log('Received chunk: ' + chunk);

    var stringChunk = String.fromCharCode.apply(null, chunk);
    cli(stringChunk);
  });

  socket.on('end', function() {
    console.log("Socket disconnected: " + socket.remoteAddress);
  });

});

server.listen(8124, function() {
  console.log("Server is listening on 8124");
});